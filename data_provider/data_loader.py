import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import h5py
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_Stocks(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S',  data_path='AAPL.csv',
                 target='OT', scale=True, timeenc=0, freq='d', percent=100,
                 seasonal_patterns=None):
        if size is None:
            raise Exception('Please specify the seq_len, label_len or pred_len variables.')
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.iloc[:, :7]

        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        total_len = len(df_raw)
        train_len = int(train_ratio * total_len)
        val_len = int(val_ratio * total_len)

        border1s = [0, train_len - self.seq_len, train_len + val_len - self.seq_len]
        border2s = [train_len, train_len + val_len, total_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Demand(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS',  data_path='demand_data_all_cleaned.csv',
                 target='actual', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, is_channel_independent=False):
        if size is None:
            raise Exception('Please specify the seq_len, label_len or pred_len variables.')
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = 0
        self.freq = freq
        self.is_channel_independent = is_channel_independent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        self.has_nems = 'forecast' in df_raw.columns

        # shift target column to the end ('MS' should be passed in from the shell script)
        reordered = [col for col in df_raw.columns if col != str(self.target)]
        reordered.append(str(self.target))
        df_raw = df_raw[reordered]

        if self.has_nems:
            # extract the forecast data; the loss calculation for this is done alongside the MOMENT model
            df_raw.drop(columns=['forecast'], inplace=True)

        if self.set_type == 0:
            print(f'[INFO]: Num features: {len(df_raw.columns)}')
            print(f'[INFO]: Dataset columns: {df_raw.columns.tolist()}')

        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2

        total_len = len(df_raw)
        train_len = int(train_ratio * total_len)
        val_len = int(val_ratio * total_len)
        test_len = int(test_ratio * total_len)

        border1s = [0, train_len - self.seq_len, train_len + val_len - self.seq_len]
        border2s = [train_len, train_len + val_len, train_len + val_len + test_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.target_scaler.fit(train_data[self.target].values.reshape(-1, 1))
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp['datetime'])
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.datetime.apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp.datetime.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.datetime.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.datetime.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.datetime.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.datetime.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['datetime'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['datetime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.is_channel_independent:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    # LSTM does not need channel independence
    def __len__(self):
        if self.is_channel_independent:
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def target_inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)


class Dataset_Carbon(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS',  data_path='merged.csv',
                 target='Price', scale=True, timeenc=0, freq='d', percent=100,
                 seasonal_patterns=None, feats_pct=None):
        if size is None:
            raise Exception('Please specify the seq_len, label_len or pred_len variables.')
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.feats_pct = feats_pct / 100 if feats_pct is not None else None # shell script can't take in floating point values

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        # self.target_scaler = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.feats_pct is not None:
            # Load in selected features based on the spearman correlation analysis
            sel_features_df0 = pd.read_excel(os.path.join(self.root_path, "ranked_abs_features_daily.xlsx"))
            sel_features_df0.sort_values(by="Correlation", ascending=False, inplace=True)
            num_features = int(len(sel_features_df0) * self.feats_pct)
            sel_feature_names = sel_features_df0["Factor"][0:num_features].tolist()
            sel_feature_names = [val for val in sel_feature_names if 'Historical Price' not in val]
            df_raw = df_raw[['Date', 'Price'] + sel_feature_names]
            assert len(df_raw.columns) == len(sel_feature_names) + 2

        df_raw = df_raw[[col for col in df_raw if col not in ['Price']] + ['Price']] # shift target to the end

        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2

        total_len = len(df_raw)
        train_len = int(train_ratio * total_len)
        val_len = int(val_ratio * total_len)
        test_len = int(test_ratio * total_len)

        border1s = [0, train_len - self.seq_len, train_len + val_len - self.seq_len]
        border2s = [train_len, train_len + val_len, train_len + val_len + test_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

            print(f'[INFO]: Dataset columns: {df_raw.columns.tolist()}')
            print(f'[INFO]: Train boundaries: {df_raw.iloc[0].Date} | {df_raw.iloc[border2s[0] - 1].Date}')
            print(f'[INFO]: Val boundaries: {df_raw.iloc[border2s[0]].Date} | {df_raw.iloc[border2s[1] - 1].Date}')
            print(f'[INFO]: Test boundaries: {df_raw.iloc[border2s[1]].Date} | {df_raw.iloc[-1].Date}')

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # df_data = np.log(df_data)  # apply log transform since data is right-skewed
            # data = df_data.values
            train_data = df_data[border1s[0]:border2s[0]]
            self.target_scaler.fit(train_data[self.target].values.reshape(-1, 1))
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp['Date'])
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp['Date'].apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp['Date'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['Date'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['Date'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['Date'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['Date'].apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(columns=['Date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
        # return np.exp(self.scaler.inverse_transform(data))
        # return np.exp(data)

    def target_inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)
        # return np.exp(self.target_scaler.inverse_transform(data))
        # return np.exp(data)


class Dataset_Carbon_Monthly(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS',  data_path='merged_data.csv',
                 target='Price', scale=True, timeenc=0, freq='m', percent=100,
                 seasonal_patterns=None, feats_pct=None):
        if size is None:
            raise Exception('Please specify the seq_len, label_len or pred_len variables.')
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.feats_pct = feats_pct / 100 if feats_pct is not None else None # shell script can't take in floating point values

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        # self.target_scaler = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.feats_pct is not None:
            # Load in selected features based on the spearman correlation analysis
            sel_features_df0 = pd.read_excel(os.path.join(self.root_path, "ranked_abs_features_monthly.xlsx"))
            sel_features_df0.sort_values(by="Correlation", ascending=False, inplace=True)
            num_features = int(len(sel_features_df0) * self.feats_pct)
            sel_feature_names = sel_features_df0["Factor"][0:num_features].tolist()
            sel_feature_names = [val for val in sel_feature_names if 'Historical Price' not in val]
            df_raw = df_raw[['Month-Year', 'Price'] + sel_feature_names]
            assert len(df_raw.columns) == len(sel_feature_names) + 2

        df_raw = df_raw[[col for col in df_raw if col not in ['Price']] + ['Price']]  # shift target to the end

        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2

        total_len = len(df_raw)
        train_len = int(train_ratio * total_len)
        val_len = int(val_ratio * total_len)
        test_len = int(test_ratio * total_len)

        border1s = [0, train_len - self.seq_len, train_len + val_len - self.seq_len]
        border2s = [train_len, train_len + val_len, train_len + val_len + test_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

            print(f'[INFO]: Dataset columns: {df_raw.columns.tolist()}')
            print(f'[INFO]: Train boundaries: {df_raw.iloc[0]["Month-Year"]} | {df_raw.iloc[border2s[0] - 1]["Month-Year"]}')
            print(f'[INFO]: Val boundaries: {df_raw.iloc[border2s[0]]["Month-Year"]} | {df_raw.iloc[border2s[1] - 1]["Month-Year"]}')
            print(f'[INFO]: Test boundaries: {df_raw.iloc[border2s[1]]["Month-Year"]} | {df_raw.iloc[-1]["Month-Year"]}')

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # data = np.log1p(df_data.values)

            # df_data = np.log1p(df_data)
            train_data = df_data[border1s[0]:border2s[0]]
            self.target_scaler.fit(train_data[self.target].values.reshape(-1, 1))
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['Month-Year']][border1:border2]
        df_stamp['Month-Year'] = pd.to_datetime(df_stamp['Month-Year'])
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp['Month-Year'].apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp['Month-Year'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['Month-Year'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['Month-Year'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['Month-Year'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['Month-Year'].apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(columns=['Month-Year']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Month-Year'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # return np.expm1(data)
        # return self.scaler.inverse_transform(np.expm1(data))
        return self.scaler.inverse_transform(data)

    def target_inverse_transform(self, data):
        # return np.expm1(data)
        # return self.scaler.inverse_transform(np.expm1(data))
        return self.target_scaler.inverse_transform(data)


class Dataset_Carbon_Daily_Decomp(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS',  data_path='merged_data.csv',
                 target='Price', scale=True, timeenc=0, freq='d', percent=100,
                 seasonal_patterns=None, feats_pct=None):
        if size is None:
            raise Exception('Please specify the seq_len, label_len or pred_len variables.')
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x)

    def __read_data__(self):
        """
        For this dataloader, no possibility of doing global scaling, since the data is already split into separate windows.
        Pass --no-scale into the script and run intra-window scaling instead.
        """

        with h5py.File(os.path.join(self.root_path, self.data_path), 'r') as f:
            decomposed_windows = f['decomposed_windows'][:]  # shape is (n_windows, channels, seq_len)
            horizons = f['horizons'][:]  # shape is (n_windows, seq_len)

        decomposed_windows = np.roll(decomposed_windows, -1, axis=1)  # shift target channel to the end
        decomposed_windows = np.transpose(decomposed_windows, (0, 2, 1))  # reshape to (n_windows, seq_len, channels)
        horizons = horizons.reshape(horizons.shape[0], horizons.shape[1], 1)  # rehape to (n_windows, pred_len, 1)

        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2

        total_len = decomposed_windows.shape[0]
        train_len = int(train_ratio * total_len)
        val_len = int(val_ratio * total_len)
        test_len = int(test_ratio * total_len)

        border1s = [0, train_len, train_len + val_len]
        border2s = [train_len, train_len + val_len, train_len + val_len + test_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            print(f'[INFO]: Number of decomposed channels: {decomposed_windows.shape[-1]}')
            print(f'Train length: {train_len}')
            print(f'Val length: {val_len}')
            print(f'Test length: {test_len}')

        self.data_x = decomposed_windows[border1:border2]
        self.data_y = horizons[border1:border2]

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark = np.zeros((self.seq_len, 1))
        seq_y_mark = np.zeros((self.pred_len, 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)
