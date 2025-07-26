import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 从当前模块导入
from .data_interface import ETTDataset as ETTDatasetInterface


class ETTDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETT-small/ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
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
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # 使用接口类读取数据
        dataset_interface = ETTDatasetInterface(self.root_path, self.data_path)
        df_raw, _ = dataset_interface.load_data()
        df_raw = dataset_interface.raw_data  # 获取包含日期的完整数据

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

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
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = self.mean
        std = self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = self.mean
        std = self.std
        return (data * std) + mean


def time_features(dates, timeenc=1, freq='h'):
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following:
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    >
    > If `timeenc` is 0:
    > * Use all these values as named columns in a dataframe.
    >
    > If `timeenc` is 1:
    > * Use sin/cos cyclical encoding for each of these temporal features.
    """
    if timeenc == 0:
        dates['month'] = dates.date.astype(object).apply(lambda row: row.month)
        dates['day'] = dates.date.astype(object).apply(lambda row: row.day)
        dates['weekday'] = dates.date.astype(object).apply(lambda row: row.weekday())
        dates['hour'] = dates.date.astype(object).apply(lambda row: row.hour)
        dates['minute'] = dates.date.astype(object).apply(lambda row: row.minute)
        dates['minute'] = dates.minute.map(lambda x: x // 15)
        freq_map = {
            'y': [], 'm': ['month'], 'w': ['month'], 'd': ['month', 'day', 'weekday'],
            'b': ['month', 'day', 'weekday'], 'h': ['month', 'day', 'weekday', 'hour'],
            't': ['month', 'day', 'weekday', 'hour', 'minute'],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        return np.transpose(time_features_from_timestamps(dates), (1, 0))


def time_features_from_timestamps(dates):
    """
    Convert timestamps to multi-dimensional time features using sine and cosine transformations
    """
    # Calculate total seconds for each timestamp relative to Unix epoch
    seconds = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    # Define the periodicity of each time component in seconds
    features = []
    
    # Yearly (approximate)
    features.append(np.sin(2 * np.pi * seconds / (365.25 * 24 * 60 * 60)))
    features.append(np.cos(2 * np.pi * seconds / (365.25 * 24 * 60 * 60)))
    
    # Monthly (approximate)
    features.append(np.sin(2 * np.pi * seconds / (30.44 * 24 * 60 * 60)))
    features.append(np.cos(2 * np.pi * seconds / (30.44 * 24 * 60 * 60)))
    
    # Weekly
    features.append(np.sin(2 * np.pi * seconds / (7 * 24 * 60 * 60)))
    features.append(np.cos(2 * np.pi * seconds / (7 * 24 * 60 * 60)))
    
    # Daily
    features.append(np.sin(2 * np.pi * seconds / (24 * 60 * 60)))
    features.append(np.cos(2 * np.pi * seconds / (24 * 60 * 60)))
    
    # Hourly
    features.append(np.sin(2 * np.pi * seconds / (60 * 60)))
    features.append(np.cos(2 * np.pi * seconds / (60 * 60)))

    return np.array(features)