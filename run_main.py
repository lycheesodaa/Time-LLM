import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin, load_checkpoint_in_model, load_checkpoint_and_dispatch
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM, Informer

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

from models.LSTMModel import LSTMModel, ConvLSTMModel, LSTMGRUModel, GRUAttentionModel, Decomp_LSTM
from models.NeuralNetwork import BPNN, CNN

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, MAPELoss

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, DLinear, TimeLLM, LSTM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default=None, help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--scale', type=bool, default=True, action=argparse.BooleanOptionalAction, help='whether to scale with the data loader')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=None, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=None, help='decoder input size')
parser.add_argument('--c_out', type=int, default=None, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--distil', type=bool, default=True)


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

parser.add_argument('--results_path', type=str, default='./results/data/')
parser.add_argument('--eval_only', type=bool, default=False)
parser.add_argument('--decomp_type', type=str, default=None)
parser.add_argument('--feats_pct', type=int, default=None)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.enc_in is None:
        args.enc_in = train_data.enc_in
        args.dec_in = train_data.enc_in
    else:
        assert args.enc_in == train_data.enc_in

    if args.c_out is None:
        args.c_out = train_data.enc_in

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'Informer':
        model = Informer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'LSTM':
        # scale within the model input window if dataloader scaling is not used
        if args.decomp_type is not None:
            model = Decomp_LSTM(args.enc_in, args.d_model, 2, args.pred_len, args.decomp_type, window_norm=not args.scale).float()
        else:
            model = LSTMModel(args.enc_in, args.d_model, 2, args.pred_len, window_norm=not args.scale).float()
    elif args.model == 'BPNN':
        model = BPNN(args.seq_len, args.enc_in, args.pred_len)
    elif args.model == 'CNN':
        model = CNN(args.seq_len, args.enc_in, args.pred_len)
    elif args.model == 'ConvLSTM':
        model = ConvLSTMModel(args.enc_in, args.d_model, 2, args.pred_len).float()
    elif args.model == 'GRU':
        model = LSTMGRUModel(args.enc_in, args.d_model, 2, args.pred_len).float()
    elif args.model == 'GRUAttention':
        model = GRUAttentionModel(args.enc_in, args.d_model, 2, args.pred_len).float()
    else:
        model = TimeLLM.Model(args).float()
        args.content = load_content(args)

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)
    if not os.path.exists(args.results_path) and accelerator.is_local_main_process:
        os.makedirs(args.results_path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, verbose=True)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)
    max_norm = 5.0

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    # mape_metric = MAPELoss

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # if os.path.exists(path + '/checkpoint'):
    #     accelerator.load_state(str(path) + '/checkpoint')
    #     vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric, best=True)
    #     exit()

    start = time.time()

    if args.eval_only:
        accelerator.load_state(str(path) + '/checkpoint')
        vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric, best=True)
        exit()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            iter_count += 1
            model_optim.zero_grad(set_to_none=True)

            batch_x = batch_x.float().to(accelerator.device) # input time horizon
            batch_y = batch_y.float().to(accelerator.device) # target horizon

            # _mark holds information about time-related features. Specifically, it is a
            # tensor that encodes temporal information and is associated with the
            # input data batch_x/batch_y. (not used)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input (not used)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate,
                    # S:univariate predict univariate, MS:multivariate predict univariate'
                    # if multivariate predict univariate',then output should be the last column of the decoder
                    # output, so f_dim = -1 to only contain the last column, else is all columns
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            # When train iters. reach a multiple of 100, print the speed, left time, loss. etc
            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            # backprop
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(model_optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        # validation and testing
        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        # test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        test_loss, test_mae_loss = 0, 0
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        # early stopping?
        early_stopping(vali_loss, model, path, (vali_loss, vali_mae_loss, test_loss, test_mae_loss))
        if early_stopping.early_stop:
            elapsed = time.time() - start
            print(f'Time elapsed: {elapsed}')

            accelerator.print("Early stopping")
            best_scores = early_stopping.all_scores
            accelerator.print(
                "Best scores | Vali Loss: {0:.7f} Test Loss: {1:.7f} MAE Loss: {2:.7f} Test MAE Loss: {3:.7f}".format(
                    *best_scores))

            accelerator.print("Exporting model predictions to CSV...")
            accelerator.load_state(str(path) + '/checkpoint')
            vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric, best=True)
            break

        # adjust learning rate
        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
# if accelerator.is_local_main_process:
    # path = './checkpoints'  # unique checkpoint saving path
    # del_files(path)  # delete checkpoint files
    # accelerator.print('success delete checkpoints')