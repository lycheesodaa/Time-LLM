import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
from sympy.printing.pretty.pretty_symbology import line_width
from torch.nn.functional import mse_loss
from tqdm import tqdm
from utils.tools import MAPELoss

save_path = 'figures/demand/'

# predlens = [96, 192, 356]
predlens = [1, 12, 72]
# dms = [160, 180, 200]
dms = [200]

# pred_len=predlens[0]
# d_model=dms[0]

for pred_len in tqdm(predlens):
    for d_model in tqdm(dms):
        # load the data and halve it (data processing accidentally concatenated two copies of the same data)
        df = pd.read_csv(f'results/data/LSTM_Demand_pl{pred_len}_dm{d_model}_predictions.csv', index_col=0)
        midpoint = len(df) // 2
        if df.iloc[:midpoint].equals(df.iloc[midpoint:].reset_index(drop=True)):
            df = df.iloc[:midpoint]
        df['date'] = pd.to_datetime(df['date'])

        if pred_len >= 12:
            # single iter of length pred_len (second pred_len horizon)
            single = df[pred_len ** 2:pred_len ** 2 + pred_len]
            min_date = single['date'].min()
            max_date = single['date'].max()
            single = single.melt(id_vars='date', value_vars=['pred', 'true'], var_name='ts', value_name='demand')
            fig = plt.figure(figsize=(12,8))
            sns.lineplot(data=single, x='date', y='demand', hue='ts')
            plt.title(f'Demand Plot (Single iteration, Horizon {pred_len})')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend(title='')
            fig.savefig(save_path + f'LSTM_Demand_pl{pred_len}_dm{d_model}_predictions_1-iter.png', bbox_inches='tight')
            plt.show()
            # exit()

            # multiple iters of the second pred_len horizon
            multiple = df.query(f"'{min_date}' <= date <= '{max_date}'")
            multiple = multiple.melt(id_vars='date', value_vars=['pred', 'true'], var_name='ts', value_name='demand')
            fig = plt.figure(figsize=(12,8))
            sns.lineplot(data=multiple, x='date', y='demand', hue='ts', errorbar='pi') # use 'sd' or 'pi'
            plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend(title='')
            fig.savefig(save_path + f'LSTM_Demand_pl{pred_len}_dm{d_model}_predictions_all-iter_pi.png', bbox_inches='tight')
            plt.show()
            # exit()

            # split-iters
            df['window'] = df.index // pred_len
            split = df.query(f"'{min_date}' <= date <= '{max_date}'")
            split = split.melt(id_vars=['date', 'window'], value_vars=['pred', 'true'], var_name='ts', value_name='demand')
            fig = plt.figure(figsize=(12,8))
            sns.lineplot(data=split, x='date', y='demand', hue='ts', style='window', linewidth=0.5, dashes=False)
            plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend([])
            fig.savefig(save_path + f'LSTM_Demand_pl{pred_len}_dm{d_model}_predictions_split.png', bbox_inches='tight')
            plt.show()
        else:
            # split-iters
            df['window'] = df.index // pred_len
            min_date = df.iloc[12]['date']
            max_date = df.iloc[12 * 2]['date']
            split = df.query(f"'{min_date}' <= date <= '{max_date}'")
            split = split.melt(id_vars=['date', 'window'], value_vars=['pred', 'true'], var_name='ts', value_name='demand')
            fig = plt.figure(figsize=(12,8))
            sns.lineplot(data=split, x='date', y='demand', hue='ts', style='window', dashes=False, markers=True)
            plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend([])
            fig.savefig(save_path + f'LSTM_Demand_pl{pred_len}_dm{d_model}_predictions_split.png', bbox_inches='tight')
            plt.show()

        # calculating losses - MSE and MAPE
        # MSE loss is 1/n * sum((y - ypred)^2)
        # MAPE is 100%/n * sum(abs((y - ypred) / y))
        batch_len = pred_len * 16
        if len(df) % batch_len != 0:
            raise Exception(f'Error with number of iterations')
        num_iters = len(df) // batch_len

        mse_losses = []
        mape_losses = []
        for itr in range(num_iters):
            data_range = df[batch_len * itr:batch_len * itr + batch_len]
            pred = torch.tensor(data_range['pred'].values)
            true = torch.tensor(data_range['true'].values)

            mse = mse_loss(pred, true).item()
            mape = MAPELoss(pred, true).item()

            mse_losses.append(mse)
            mape_losses.append(mape)

        df_losses = pd.DataFrame({
            'mse': mse_losses,
            'mape': mape_losses
        })

        # df_losses = pd.read_csv(f'results/data/LSTM_Demand_pl96_dm{d_model}_losses.csv')
        fig = plt.figure(figsize=(12,8))
        sns.lineplot(data=df_losses, x=df_losses.index, y='mse', label='mse_loss', color='red')
        plt.title('Loss Plot')
        plt.xlabel('Batch')
        plt.ylabel('MSE Loss')
        plt.legend(title='Time Series')
        fig.savefig(save_path + f'LSTM_Demand_pl{pred_len}_dm{d_model}_loss_mse.png', bbox_inches='tight')
        plt.show()

        fig = plt.figure(figsize=(12,8))
        sns.lineplot(data=df_losses, x=df_losses.index, y='mape', label='mape_loss', color='red')
        plt.title('Loss Plot')
        plt.xlabel('Batch')
        plt.ylabel('Percentage Loss')
        plt.legend(title='Time Series')
        fig.savefig(save_path + f'LSTM_Demand_pl{pred_len}_dm{d_model}_loss_mape.png', bbox_inches='tight')
        plt.show()