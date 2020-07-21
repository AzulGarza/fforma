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
from ESRNN.utils_evaluation import owa

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

    def prepare_datasets_for_model(self, X_df, padded_df, horizons):
        # Reshape to tensor
        n_series = len(horizons)
        max_horizon = max(horizons)
        n_models = padded_df.columns.size - 3 #2 por unique_id y ds, TODO: sacar hardcodeo
        
        y = padded_df['y'].values.reshape((n_series, max_horizon))
        padded_df = padded_df.drop(['unique_id','ds','y'], axis=1)
        preds = padded_df.values.reshape((n_series, max_horizon, n_models))
        horizons = np.expand_dims(horizons,1)

        # Send datasets to torch tensors
        y = torch.tensor(y, dtype=torch.float32)
        preds = torch.tensor(preds, dtype=torch.float32)
        horizons = torch.tensor(horizons, dtype=torch.float32)

        X_df = X_df.set_index(['unique_id'])
        X = X_df.values

        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        return X, y, preds, horizons, max_horizon, n_models

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
        
        # Pad predictions and y dataframes
        preds_df = preds_df.merge(y_df, on=['unique_id','ds'], how='outer')
        padded_df, horizons = self.pad_long_df(preds_df)

        X, y, preds, horizons, max_horizon, n_models = self.prepare_datasets_for_model(X_df, padded_df, horizons)
        self.horizons = horizons
        self.max_horizon = max_horizon
        self.n_models = n_models

        loss = self.params['loss_function']

        self.model = NeuralNetwork(num_numerical_cols=X.size()[1],
                                   output_size=n_models,
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

        train_data = torch.utils.data.TensorDataset(X, y, preds, horizons)
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
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        # Forecast
        with torch.no_grad():
            self.model.eval()
            forecast = self.model(X, preds)

        y_hat_padded = forecast.data.numpy().flatten()

        # Despadeo
        y_hat_padded = y_hat_padded.reshape((n_series, self.max_horizon))
        y_hat = []
        for i in range(n_series):
            y_hat_i = y_hat_padded[i, :pred_horizons[i]]
            y_hat.append(y_hat_i)
        y_hat = np.hstack(y_hat)
        
        return y_hat
