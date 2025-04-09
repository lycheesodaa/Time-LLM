import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.all_scores = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path, all_scores=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.all_scores = all_scores
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.all_scores = all_scores
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            self.accelerator.save_state(path + '/checkpoint', safe_serialization=False)
            # torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, best=False):
    total_loss = []
    total_mae_loss = []
    all_preds = []
    all_true = []
    # dates = []

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(vali_loader), total=len(vali_loader)):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            # print('x:', batch_x.shape)
            # print('y', batch_y.shape)

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y, batch_y_mark = accelerator.gather_for_metrics((outputs, batch_y, batch_y_mark))

            # print('outputs:', outputs.shape)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)

            if best:
                all_preds.append(pred.reshape(-1, 1))
                all_true.append(true.reshape(-1, 1))
                # dates.append(batch_y_mark.detach()[:, -args.pred_len:, :].view(-1, 6))

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    if best:
        # Concatenate all predictions and indices
        all_preds = accelerator.gather(torch.cat(all_preds))
        all_true = accelerator.gather(torch.cat(all_true))
        # dates = accelerator.gather(torch.cat(dates))

        if accelerator.is_main_process:
            all_preds = all_preds.cpu().float().numpy()
            all_true = all_true.cpu().float().numpy()
            # dates = dates.cpu().float().numpy()

            if args.scale:
                # inverse the scaling
                all_preds = vali_data.target_inverse_transform(all_preds.reshape(-1, 1)).reshape(-1)
                all_true = vali_data.target_inverse_transform(all_true.reshape(-1, 1)).reshape(-1)

            df = pd.DataFrame({
                'pred': all_preds.reshape(-1),
                'true': all_true.reshape(-1)
            })

            # extra line just for decomposition LSTM models
            model_name = args.model if args.decomp_type is None else f'{args.decomp_type}-{args.model}'

            export_str = args.results_path + f'{model_name}_{args.data}_pl{args.pred_len}_dm{args.d_model}_predictions.csv'
            df.to_csv(export_str)

            log_into_csv(
                results_df=df,
                name=args.des,
                stage='',
                model=model_name,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                lr=args.learning_rate,
                bsz=args.batch_size,
                log_file_name=f'{args.data}_{model_name}',
                pred_col_name='pred'
            )

            print(f"Exported to {export_str}")

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def create_datetime(row):
    row = row.astype(int)
    return datetime(row['year'], row['month'], row['day'], row['hour'], row['minute'])


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


def MAPELoss(pred, true):
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    Args:
    pred (torch.Tensor): Predicted values
    true (torch.Tensor): True values

    Returns:
    torch.Tensor: MAPE value
    """
    epsilon = 1e-8  # Small constant to avoid division by zero
    return torch.mean(torch.abs((true - pred) / (true + epsilon))) * 100


def calculate_mse(y_true: list, y_pred: list) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Args:
    y_true: Array of true values
    y_pred: Array of predicted values

    Returns:
    float: The calculated MSE

    Raises:
    ValueError: If the input arrays have different shapes
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted arrays must have the same shape")

    return np.mean((y_true - y_pred) ** 2)


def calculate_mape(y_true: list, y_pred: list) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Args:
    y_true: Array of true values
    y_pred: Array of predicted values

    Returns:
    float: The calculated MAPE

    Raises:
    ValueError: If the input arrays have different shapes
    ZeroDivisionError: If any true value is zero
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted arrays must have the same shape")

    if np.any(y_true == 0):
        raise ZeroDivisionError("MAPE is undefined when true values contain zeros")

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def log_into_csv(
    results_df: pd.DataFrame,
    name: str,
    stage: str,
    model: str = 'LSTM',
    seq_len: int = 512,
    pred_len: int = 96,
    lr: float = None,
    bsz: int = 16,
    log_file_name: str = 'demand',
    pred_col_name: str = 'actual',
    args: dict | None = None,
):
    log_file = f'results/{log_file_name}_runs.csv'

    # Create sample first line in records
    if not os.path.exists(log_file):
        df = pd.DataFrame({
            'timestamp': datetime.now(),
            'name': 'sample',
            'stage': 'finetuned',
            'model': 'LSTM',
            'seq_len': 512,
            'pred_len': 96,
            'lr': 0.01,
            'bsz': 16,
            'score_type': 'mape',
            'score': 1.23,
        }, index=[0])
        df.to_csv(log_file)

    curr_run = pd.DataFrame({
        'timestamp': datetime.now(),
        'name': name,
        'stage': stage,
        'model': model,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'lr': lr,
        'bsz': bsz,
        'score_type': 'mape',
        'score': calculate_mape(results_df['true'], results_df[pred_col_name])
    }, index=[0])

    df = pd.read_csv(log_file, index_col=0)
    assert len(df.columns) == len(curr_run.columns)

    df = pd.concat([df, curr_run]).reset_index(drop=True)
    df.to_csv(log_file)
