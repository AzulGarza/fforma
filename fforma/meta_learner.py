#!/usr/bin/env python
# coding: utf-8

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb
import itertools

from copy import deepcopy
from tqdm import tqdm
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

from fforma.metrics import SMAPE1Loss

softmax = nn.Softmax(1)


class MetaLearner(object):
    """

    """
    def __init__(self, params, contribution_to_error, random_seed=1):
        self.params = params
        self.contribution_to_error = contribution_to_error
        self.random_seed = random_seed

    def fobj(self, predt, dtrain):
        """
        """
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')
        preds_transformed = softmax(preds, axis=1)

        weighted_avg_loss_func = (preds_transformed * self.contribution_to_error[y, :]).sum(axis=1).reshape((n_train, 1))

        grad = preds_transformed * (self.contribution_to_error[y, :] - weighted_avg_loss_func)
        hess = self.contribution_to_error[y,:] * preds_transformed * (1.0 - preds_transformed) - grad * preds_transformed
        #hess = grad*(1 - 2*preds_transformed)
        return grad.flatten('F'), hess.flatten('F')

    def feval(self, predt, dtrain):
        """
        """
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')
        preds_transformed = softmax(preds, axis=1)
        weighted_avg_loss_func = (preds_transformed * self.contribution_to_error[y, :]).sum(axis=1)
        fforma_loss = weighted_avg_loss_func.mean()

        return 'FFORMA-loss', fforma_loss, False

    def fit(self, features, best_models, early_stopping_rounds, verbose_eval):
        """
        """
        feats_train, \
            feats_val, \
            best_models_train, \
            best_models_val, \
            indices_train, \
            indices_val = train_test_split(features,
                                           best_models,
                                           np.arange(features.shape[0]),
                                           random_state=self.random_seed,
                                           stratify=best_models)

        params = deepcopy(self.params)
        num_round = int(params.pop('n_estimators', 100))

        params['num_class'] = len(np.unique(best_models))

        dtrain = lgb.Dataset(data=feats_train, label=indices_train)
        dvalid = lgb.Dataset(data=feats_val, label=indices_val)
        valid_sets = [dtrain, dvalid]

        self.gbm_model = lgb.train(
            params=params,
            train_set=dtrain,
            fobj=self.fobj,
            num_boost_round=num_round,
            feval=self.feval,
            valid_sets=valid_sets,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )

    def predict(self, features, tmp=1):
        """
        """
        scores = self.gbm_model.predict(features, raw_score=True)
        weights = softmax(scores / tmp, axis=1)
        return weights

##############################################################################
################### CUSTOM
##############################################################################

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
        forecast = torch.einsum('ij,ikj->ik', theta, v)
        return forecast

class MetaLearnerNN(object):
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
    def __init__(self, params, random_seed=1):
        self.params = deepcopy(params)
        self.use_softmax = self.params.pop('use_softmax', False)
        self.random_seed = random_seed

    def parse_datasets(self, preds_df, y_df):

        horizons = preds_df.groupby('unique_id')['ds'].count().values
        max_horizon = max(horizons)

        # Padding
        dict_df = {'unique_id':preds_df['unique_id'].unique(),
                'ds':list(range(1, max_horizon+1))}
        padding_dict = list(itertools.product(*list(dict_df.values())))
        padding_dict = pd.DataFrame(padding_dict, columns=list(dict_df.keys()))
        df_padded = padding_dict.merge(preds_df, on=['unique_id','ds'], how='outer')
        df_padded = df_padded.merge(y_df, on=['unique_id','ds'], how='outer')
        df_padded = df_padded.fillna(0)
        df_padded = df_padded.sort_values(['unique_id','ds']).reset_index(drop=True)

        # Reshape to tensor
        n_series = len(horizons)
        n_models = preds_df.columns.size - 2 #2 por unique_id y ds, TODO: sacar hardcodeo

        y = df_padded['y'].values.reshape((n_series, max_horizon))
        df_padded = df_padded.drop(['unique_id','ds','y'], axis=1)
        preds = df_padded.values.reshape((n_series, max_horizon, n_models))
        horizons = np.expand_dims(horizons,1)

        return y, preds, horizons, n_series, n_models

    def fit(self, X_df, preds_df, y_df, verbose_eval=True):
        """
        Parameters
        ----------
        features: numpy array
            Numpy array of size N * f.
        best_models: numpy array
            Numpy array of size N.
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        y, preds, horizons, n_series, n_models = self.parse_datasets(preds_df, y_df)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.preds = torch.tensor(preds, dtype=torch.float32)
        self.horizons = torch.tensor(horizons, dtype=torch.float32)
        self.n_series = n_series
        self.n_models = n_models

        X_df = X_df.set_index(['unique_id'])
        X = X_df.values
        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        loss = self.params['loss_function']

        self.model = NeuralNetwork(num_numerical_cols=X_df.shape[1],
                                   output_size=self.n_models,
                                   layers=self.params['layers'],
                                   p=self.params['dropout'],
                                   use_softmax=self.use_softmax)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     betas=(0.9, 0.999),
                                     lr=self.params['learning_rate'],
                                     eps=self.params['gradient_eps'],
                                     weight_decay=self.params['weight_decay'])

        lr_scheduler = StepLR(optimizer=optimizer,
                              step_size=self.params['lr_scheduler_step_size'],
                              gamma=self.params['lr_decay'])

        train_data = torch.utils.data.TensorDataset(X, self.y, self.preds, self.horizons)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.params['batch_size'])

        for epoch in range(self.params['epochs']):
            self.model.train()
            start = time.time()
            epoch_losses = []
            for i, data in enumerate(train_loader):
                batch_x, batch_y, batch_preds, batch_h = data
                batch_weights = 1/batch_h

                optimizer.zero_grad()
                batch_y_hat = self.model(batch_x, batch_preds)
                train_loss = loss(batch_y, batch_y_hat, batch_weights)
                train_loss.backward()
                optimizer.step()

                epoch_losses.append(train_loss.data.numpy())

            # Decay learning rate
            lr_scheduler.step()

            self.train_loss = np.mean(epoch_losses)

            if verbose_eval:
                print(f"========= Epoch {epoch} finished =========")
                print(f"Training time: {round(time.time() - start, 5)}")
                print(f"Training loss: {self.train_loss:.5f}")

        return self

    def predict(self, X_df, preds_df, y_df, tmp=1):
        """
        """
        y_hat_df = y_df[['unique_id', 'ds', 'y', 'y_hat_naive2']]
        y_hat_df = y_hat_df.sort_values(['unique_id','ds']).reset_index(drop=True)

        y_df = y_df.drop(['y_hat_naive2'], axis=1)
        _, preds_test, _, _, _ = self.parse_datasets(preds_df, y_df)

        preds_test = torch.tensor(preds_test, dtype=torch.float32)

        self.model.eval()

        X_df = X_df.set_index(['unique_id'])
        X = X_df.values
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        forecast = self.model(X, preds_test)
        y_hat_df['y_hat'] = forecast.data.numpy().flatten()

        return y_hat_df
