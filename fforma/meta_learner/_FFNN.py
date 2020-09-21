#!/usr/bin/env python
# coding: utf-8

import time
import torch
import itertools
import torch as t
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing as mp
import xgboost as xgb

from copy import deepcopy
from tqdm import tqdm
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

from tsfeatures.metrics import evaluate_panel
from src.meta_evaluation import calc_errors_widing
from src.metrics.metrics import mape, smape


LOSS_DICT = {'smape': smape,
             'mape': mape}

class NeuralNetwork(nn.Module):

    def __init__(self, num_numerical_cols, output_size, layers, p=0.4,
                 use_softmax=False):
        super().__init__()
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        input_size = num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        if use_softmax:
            all_layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x, v):
        theta = self.layers(x)
        # print(x.shape)
        # print(theta.shape)
        # print(v.shape)
        forecast = torch.einsum('ij,ikj->ik', theta, v)
        #print(forecast.shape)
        return forecast

class MetaLearnerFFNN(object):
    """Evaluates ensemble model on the fly using neural networks.

    Parameters
    ----------
    actual_y: numpy array
        Actual values of the time series.
        Numpy array of size N * h
    preds_y_val: numpy array
        Model predictions to ensemble.
        Numpy array of size N * h * m.
    h: int
        Horizon of the validation set.
    weights: numpy array
        Weighted errors.
    loss_function: pytorch loss function

    random_seed:

    """
    def __init__(self, params):
        self.params = deepcopy(params)
        self.use_softmax = self.params.pop('use_softmax', False)

    def to_device(self, x):
        return x.to(self.params['device'])

    def pad_long_df(self, long_df):
        horizons = long_df.groupby('unique_id')['ds'].count().values
        max_horizon = max(horizons)

        dict_df = {'unique_id': long_df['unique_id'].unique(),
                   'ds': list(range(1, max_horizon+1))}
        padding_dict = list(itertools.product(*list(dict_df.values())))
        padding_dict = pd.DataFrame(padding_dict, columns=list(dict_df.keys()))
        padded_df = padding_dict.merge(long_df, on=['unique_id','ds'], how='outer')
        padded_df = padded_df.fillna(0)
        padded_df = padded_df.sort_values(['unique_id','ds']).reset_index(drop=True)
        return padded_df, horizons

    def prepare_datasets_for_model(self, X_df, padded_df, horizons, train):
        # Reshape to tensor
        n_series = len(horizons)
        max_horizon = max(horizons)
        n_models = padded_df.columns.size - 3 #2 por unique_id y ds, TODO: sacar hardcodeo

        y = padded_df['y'].values.reshape((n_series, max_horizon))
        padded_df = padded_df.drop(['unique_id','ds','y'], axis=1)
        preds = padded_df.values.reshape((n_series, max_horizon, n_models))
        horizons = np.expand_dims(horizons, 1)

        masks = np.zeros((n_series, max_horizon))
        for idx, h in enumerate(horizons):
            masks[idx, :h.item()] = 1

        y = torch.tensor(y, dtype=torch.float32)
        preds = torch.tensor(preds, dtype=torch.float32)
        horizons = torch.tensor(horizons, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        X_df = X_df.set_index(['unique_id'])
        X = X_df.values

        # if train:
        #     self.scaler = StandardScaler().fit(X)
        #     X = self.scaler.transform(X)
        # else:
        #     X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        return X, y, preds, horizons, masks, max_horizon, n_models

    def evaluate_performance(self, dataloader, metric):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for batch in dataloader:
                batch_x, batch_y, batch_preds, batch_h, batch_mask = map(self.to_device, batch)

                batch_y_hat = self.model(batch_x, batch_preds)
                loss = LOSS_DICT[metric](batch_y.data.numpy(),
                                         batch_y_hat.data.numpy(),
                                         batch_mask.data.numpy())

                losses.append(loss)
        avg_loss = np.mean(losses)
        return avg_loss

    def fit(self, X_df, preds_df, y_df,
            X_df_test=None, preds_df_test=None, y_df_test=None,
            verbose=True):
        """
        Parameters
        ----------
        features: numpy array
            Numpy array of size N * f.
        best_models: numpy array
            Numpy array of size N.
        """
        torch.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])

        # Pad predictions and y dataframes
        preds_df = preds_df.merge(y_df, on=['unique_id','ds'], how='outer')

        # for col in preds_df.columns.difference(['unique_id', 'ds', 'naive_forec']):
        #     preds_df[col] /= preds_df['naive_forec']
        # preds_df['naive_forec'] /= preds_df['naive_forec']

        padded_df, horizons = self.pad_long_df(preds_df)

        X, y, preds, horizons, masks, max_horizon, n_models = self.prepare_datasets_for_model(X_df, padded_df, horizons, train=True)
        self.horizons = horizons
        self.max_horizon = max_horizon
        self.n_models = n_models

        #preprocessing test set
        if X_df_test is not None:
            preds_df_test = preds_df_test.merge(y_df_test, on=['unique_id','ds'], how='outer')
            padded_df_test, horizons_test = self.pad_long_df(preds_df_test)

            X_test, y_test, preds_test, horizons_test, masks_test, *_ = self.prepare_datasets_for_model(X_df_test, padded_df_test, horizons_test, train=False)

        #SETTING model
        loss = self.params['loss_function']

        self.model = NeuralNetwork(num_numerical_cols=X.size()[1],
                                   output_size=n_models,
                                   layers=self.params['layers'],
                                   p=self.params['dropout'],
                                   use_softmax=self.use_softmax)
        self.model = self.to_device(self.model)

        optimizer = torch.optim.Adam(self.model.parameters(),
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

        train_data = torch.utils.data.TensorDataset(X, y, preds, horizons, masks)
        train_loader = torch.utils.data.DataLoader(train_data, self.params['batch_size'], shuffle=True)

        if X_df_test is not None:
            test_data = torch.utils.data.TensorDataset(X_test, y_test, preds_test, horizons_test, masks_test)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)

        self.test_min_smape = 100.0
        self.test_min_mape = 100.0
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
                    break

                train_loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(train_loss.cpu().data.numpy())

            # Decay learning rate
            #lr_scheduler.step()
            # for param_group in optimizer.param_groups:
            #    param_group['lr'] = self.params['lr'] * 0.5 ** (epoch // lr_decay_step)

            self.train_loss = np.mean(epoch_losses)

            if ((epoch + 1) % self.params['display_step']) == 0:
                if verbose:
                    print("Epoch:", '%d,' % (epoch + 1),
                          "Time: {:03.3f},".format(time.time()-start),
                          "Loss: {:.4f},".format(self.train_loss))

                if X_df_test is not None:
                    test_mape = self.evaluate_performance(test_loader, 'mape')
                    test_smape = self.evaluate_performance(test_loader, 'smape')

                    if test_mape < self.test_min_mape:
                        self.test_min_mape = test_mape

                    if test_smape < self.test_min_smape:
                        self.test_min_smape = test_smape

                    if verbose:
                        print("Test SMAPE: {:.4f},".format(test_smape),
                              "Test MAPE: {:.4f}".format(test_mape))

        return self

    def predict(self, X_df, preds_df):
        """
        """
        n_series = len(X_df)
        preds_df = preds_df.sort_values(['unique_id','ds']).reset_index(drop=True)
        pred_horizons = preds_df.groupby('unique_id')['ds'].count().values

        # Prepare test data
        padded_df, _ = self.pad_long_df(preds_df)
        padded_df = padded_df.drop(['unique_id','ds'], axis=1)
        preds = padded_df.values.reshape(n_series, self.max_horizon, self.n_models)
        preds = torch.tensor(preds, dtype=torch.float32)

        X_df = X_df.set_index(['unique_id'])
        X = X_df.values
        #X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        X, preds = map(self.to_device, [X, preds])

        # Forecast
        with torch.no_grad():
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

        return y_hat
