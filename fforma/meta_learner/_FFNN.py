#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import time
from typing import List
from itertools import product

import torch as t
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import StepLR


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = t.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class FeedForwardNeuralNetwork(nn.Module):

    def __init__(self, num_numerical_cols: int,
                 output_size: int,
                 layers: List,
                 dropout: float,
                 activation: str,
                 use_softmax: bool = False,
                 initialization: str = 'xavier'):
        super().__init__()
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        input_size = num_numerical_cols

        if activation == 'relu':
            activation_fun = nn.ReLU
        elif activation == 'selu':
            activation_fun = nn.SELU

        def weights_init(m):
            if isinstance(m, nn.Linear):
                if initialization == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                    #nn.init.constant_(m.bias, 0)
                elif initialization == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    #nn.init.zero_(m.bias)

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(activation_fun())
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(dropout))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        if use_softmax:
            all_layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*all_layers)
        self.layers = self.layers.apply(weights_init)

    def forward(self, x, v):
        theta = self.layers(x)
        forecast = t.einsum('ij,ikj->ik', theta, v)

        return forecast

class MetaLearnerFFNN(object):
    """Evaluates ensemble model on the fly using neural networks.

    Parameters
    ----------
    """
    def __init__(self, params):
        self.params = deepcopy(params)

        self.models = None

    def to_device(self, x):
        return x.to(self.params['device'])

    def pad_long_df(self, long_df):
        horizons = long_df.groupby('unique_id', sort=False)['ds'].count().values
        max_horizon = max(horizons)

        dict_df = {'unique_id': long_df['unique_id'].unique(),
                   'ds': list(range(1, max_horizon + 1))}

        padding_dict = list(product(*list(dict_df.values())))
        padding_dict = pd.DataFrame(padding_dict, columns=list(dict_df.keys()))

        padded_df = padding_dict.merge(long_df, on=['unique_id','ds'], how='outer')
        padded_df = padded_df.fillna(0)
        #padded_df = padded_df.sort_values(['unique_id','ds']).reset_index(drop=True)

        return padded_df, horizons

    def prepare_datasets_for_model(self, X_df, padded_df, horizons):
        # Reshape to tensor
        n_series = len(horizons)
        max_horizon = max(horizons)
        n_models = padded_df.columns.size - 3 #2 por unique_id y ds, TODO: sacar hardcodeo

        y = padded_df['y'].values.reshape((n_series, max_horizon))
        padded_df = padded_df.drop(['unique_id', 'ds', 'y'], axis=1)
        preds = padded_df.values.reshape((n_series, max_horizon, n_models))
        horizons = np.expand_dims(horizons, 1)

        masks = np.zeros((n_series, max_horizon))
        for idx, h in enumerate(horizons):
            masks[idx, :h.item()] = 1

        y = t.tensor(y, dtype=t.float32)
        preds = t.tensor(preds, dtype=t.float32)
        horizons = t.tensor(horizons, dtype=t.float32)
        masks = t.tensor(masks, dtype=t.float32)

        X = X_df.set_index('unique_id').values

        if self.params['scaler'] == 'min_max':
            self.scaler = MinMaxScaler().fit(X)
        elif self.params['scaler'] == 'standard':
            self.scaler = StandardScaler().fit(X)
        elif self.params['scaler'] == 'no':
            self.scaler = FunctionTransformer().fit(X)
        X = self.scaler.transform(X)

        X = t.tensor(X, dtype=t.float32)

        return X, y, preds, horizons, masks, max_horizon, n_models

    def fit(self, X_df, preds_df, y_df):
        """
        Parameters
        ----------
        """
        t.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])

        # Normalizing preds and actual value
        benchmark = self.params['benchmark']
        self.models = preds_df.columns.difference(['unique_id', 'ds', benchmark])
        self.models = self.models.to_list()
        preds_df = preds_df.merge(y_df, on=['unique_id', 'ds'], how='outer')
        preds_df['ds'] = preds_df.groupby('unique_id').cumcount() + 1

        if self.params['scale_y']:
            for col in self.models + ['y']:
                preds_df[col] /= preds_df[benchmark]
        preds_df = preds_df[['unique_id', 'ds', 'y'] + self.models]

        # Pad predictions and y dataframes

        padded_df, horizons = self.pad_long_df(preds_df)


        X, y, preds, horizons, masks, max_horizon, n_models = self.prepare_datasets_for_model(X_df, padded_df, horizons)
        self.horizons = horizons
        self.max_horizon = max_horizon
        self.n_models = n_models

        #SETTING model
        loss = self.params['loss_function']

        self.model = FeedForwardNeuralNetwork(num_numerical_cols=X.size()[1],
                                              output_size=n_models,
                                              layers=self.params['layers'],
                                              dropout=self.params['dropout'],
                                              activation=self.params['activation'],
                                              use_softmax=self.params['softmax'],
                                              initialization=self.params['initialization'])
        self.model = self.to_device(self.model)

        optimizer = t.optim.Adam(self.model.parameters(),
                                 #betas=(0.9, 0.999),
                                 lr=self.params['lr'])
                                 #eps=self.params['gradient_eps'],
                                 #weight_decay=self.params['weight_decay'])

        # lr_scheduler = StepLR(optimizer=optimizer,
        #                       step_size=self.params['lr_scheduler_step_size'],
        #                       gamma=self.params['lr_decay'])

        lr_decay_step = self.params['n_epochs'] // 3
        if lr_decay_step == 0:
           lr_decay_step = 1

        train_data = t.utils.data.TensorDataset(X, y, preds, horizons, masks)
        train_loader = FastTensorDataLoader(X, y, preds, horizons, masks,
                                            batch_size=self.params['batch_size'],
                                            shuffle=True)

        for epoch in range(self.params['n_epochs']):
            self.model.train()
            start = time.time()
            epoch_losses = []

            for i, data in enumerate(train_loader):
                batch_x, batch_y, batch_preds, batch_h, batch_mask = map(self.to_device, data)

                optimizer.zero_grad()

                batch_y_hat = self.model(batch_x, batch_preds)
                train_loss = loss(batch_y, batch_y_hat, batch_mask)

                if np.isnan(float(train_loss)):
                    raise Exception('NAN loss')

                train_loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(train_loss.cpu().data.numpy())

            # Decay learning rate
            #lr_scheduler.step()
            for param_group in optimizer.param_groups:
               param_group['lr'] = self.params['lr'] * 0.5 ** (epoch // lr_decay_step)

            self.train_loss = np.mean(epoch_losses)

            if ((epoch + 1) % self.params['display_step']) == 0:
                if self.params['verbose']:
                    print("Epoch:", '%d,' % (epoch + 1),
                          "Time: {:03.3f},".format(time.time()-start),
                          "Loss: {:.4f},".format(self.train_loss))

        return self

    def predict(self, X_df, preds_df):
        """
        """
        n_series = len(X_df)
        #preds_df = preds_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        preds_df = preds_df.copy()
        preds_df = preds_df[['unique_id', 'ds'] + self.models]
        y_hat_df = preds_df[['unique_id', 'ds']].copy()

        preds_df['ds'] = preds_df.groupby('unique_id').cumcount() + 1
        pred_horizons = preds_df.groupby('unique_id', sort=False)['ds'].count().values

        # Prepare test data
        padded_df, _ = self.pad_long_df(preds_df)
        padded_df = padded_df.drop(['unique_id','ds'], axis=1)
        preds = padded_df.values.reshape(n_series, self.max_horizon, self.n_models)
        preds = t.tensor(preds, dtype=t.float32)

        X = X_df.set_index('unique_id').values
        X = self.scaler.transform(X)

        X = t.tensor(X, dtype=t.float32)

        X, preds = map(self.to_device, [X, preds])

        # Forecast
        with t.no_grad():
            self.model.eval()
            forecast = self.model(X, preds)

        y_hat_padded = forecast.cpu().data.numpy().flatten()

        # Despadeo
        y_hat_padded = y_hat_padded.reshape((n_series, self.max_horizon))
        y_hat = []
        for i in range(n_series):
            y_hat_i = y_hat_padded[i, :pred_horizons[i]]
            y_hat.append(y_hat_i)
        y_hat = np.hstack(y_hat)

        y_hat_df['y_hat'] = y_hat

        return y_hat_df
