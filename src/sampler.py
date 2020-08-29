import numpy as np
import pandas as pd
import torch as t
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class TimeseriesDataset(Dataset):
    def __init__(self,
                 model: str,
                 ts_data: list,
                 static_data: list,
                 meta_data: list,
                 offset:int,
                 window_sampling_limit: int,
                 input_size: int,
                 output_size: int,
                 batch_size: int):
        """
        model
        """
        self.model = model
        self.n_series = len(ts_data)
        self.window_sampling_limit = window_sampling_limit
        self.input_size = input_size
        self.output_size = output_size
        self.max_len = max([len(ts['y']) for ts in ts_data])
        self.n_channels = ts_data[0].values.shape[1] #len(ts_data[0].values)
        self.time_series, self.len_series = self.create_tensor(ts_data)
        self.meta_data = meta_data
        self.static_data = static_data
        self.batch_size = batch_size
        self.update_offset(offset)
        self._is_train = True

    def update_offset(self, offset):
        self.offset = offset

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False

    def get_meta_data_var(self, var):
        var_values = [x[var] for x in self.meta_data]
        return var_values

    def create_tensor(self, ts_data):
        ts_tensor = np.zeros((self.n_series, self.n_channels, self.max_len))
        len_series = []
        for idx in range(self.n_series):
            ts_idx = ts_data[idx].values.T#np.array(list(ts_data[idx].values))
            # print(ts_idx)
            # print(ts_idx.shape)
            ts_tensor[idx, :, -ts_idx.shape[1]:] = ts_idx
            len_series.append(ts_idx.shape[1])
        return ts_tensor, np.array(len_series)

    def __len__(self):
        return len(self.len_series)

    def __iter__(self):
        while True:
            if self._is_train:
                sampled_ts_indices = np.random.randint(self.n_series, size=self.batch_size)
            else:
                sampled_ts_indices = range(self.n_series)

            batch_dict = defaultdict(list)
            for index in sampled_ts_indices:
                batch_i = self[index]
                for key in batch_i:
                    batch_dict[key].append(batch_i[key])

            batch = defaultdict(list)
            for key in batch_dict:
                batch[key] = np.stack(batch_dict[key])

            # for key in batch:
            #     print(key)
            #     print(batch[key].shape)
            #     print(batch[key])

            yield batch

    def __getitem__(self, index):
        if self.model == 'nbeats':
            return self.nbeats_batch(index)
        if self.model == 'qfforma':
            return self.qfforma_batch(index)
        elif self.model == 'esrnn':
            assert 1<0, 'hacer esrnn'
        else:
            assert 1<0, 'error'

    def nbeats_batch(self, index):
        insample = np.zeros((self.n_channels, self.input_size))
        insample_mask = np.zeros(self.input_size)
        outsample = np.zeros((self.n_channels, self.output_size))
        outsample_mask = np.zeros(self.output_size)

        ts = self.time_series[index]
        len_ts = self.len_series[index]
        init_ts = max(self.max_len-len_ts+1, self.max_len-self.offset-self.window_sampling_limit)

        assert self.max_len-self.offset > init_ts, f'Offset too big for serie {index}'
        if self._is_train:
            cut_point = np.random.randint(low=init_ts,
                                          high=self.max_len-self.offset, size=1)[0]
        else:
            cut_point = max(self.max_len-self.offset, self.input_size)

        insample_window = ts[:, max(0, cut_point - self.input_size):cut_point]
        insample_mask_start = min(self.input_size, cut_point - init_ts+1) #TODO: por comentar
        insample[:, -insample_window.shape[1]:] = insample_window
        insample_mask[-insample_mask_start:] = 1.0

        outsample_window = ts[:, cut_point:min(self.max_len, cut_point + self.output_size)]
        outsample[:, :outsample_window.shape[1]] = outsample_window
        outsample_mask[:outsample_window.shape[1]] = 1.0

        insample_y = insample[0, :]
        insample_x_t = insample[1:, :]

        outsample_y = outsample[0, :]
        outsample_x_t = outsample[1:, :]

        sample = {'insample_y':insample_y, 'insample_x_t':insample_x_t, 'insample_mask':insample_mask,
                  'outsample_y':outsample_y, 'outsample_x_t':outsample_x_t, 'outsample_mask':outsample_mask}

        return sample

    def qfforma_batch(self, index):
        insample = np.zeros((self.n_channels, self.input_size))
        insample_mask = np.zeros(self.input_size)
        outsample = np.zeros((self.n_channels, self.output_size))
        outsample_mask = np.zeros(self.output_size)

        ts = self.time_series[index]
        len_ts = self.len_series[index]
        init_ts = max(self.max_len-len_ts+1, self.max_len-self.offset-self.window_sampling_limit)

        assert self.max_len-self.offset > init_ts, f'Offset too big for serie {index}'
        if self._is_train:
            cut_point = np.random.randint(low=init_ts,
                                          high=self.max_len-self.offset, size=1)[0]
        else:
            cut_point = max(self.max_len-self.offset, self.input_size)

        insample_window = ts[:, max(0, cut_point - self.input_size):cut_point]
        insample_mask_start = min(self.input_size, cut_point - init_ts+1) #TODO: por comentar
        insample[:, -insample_window.shape[1]:] = insample_window
        insample_mask[-insample_mask_start:] = 1.0

        outsample_window = ts[:, cut_point:min(self.max_len, cut_point + self.output_size)]
        outsample[:, :outsample_window.shape[1]] = outsample_window
        outsample_mask[:outsample_window.shape[1]] = 1.0

        insample_y = insample[0, :]
        static_data = self.static_data[index]

        outsample_y = outsample[0, :]



        sample = {'insample_y':insample_y, 'statitc_data': static_data, 'insample_mask':insample_mask,
                  'outsample_y':outsample_y, 'outsample_mask':outsample_mask}

        return sample
