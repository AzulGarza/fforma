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

##############################################################################
# FFORMA
##############################################################################

class MetaLearnerXGBoost(object):
    """Feature-based Forecast Model Averaging (FFORMA).


    Parameters
    ----------
    params: dict
        Dictionary of paratemeters to train gradient boosting.
    h: int
        Forecast horizon.
    seasonality: int
        Time series frequency.
    base_models: dict
        Dictionary of models to train. Ej {'ARIMA': ARIMA()}.
        Default None. Models: 'SeasonalNaive', 'Naive2', 'RandomWalkDrift'.
    metric: str
        Metric used to calculate contribution to error.
        By default 'owa' is used.
    early_stopping_rounds: int
        Maximum number of gradient boosting rounds.
        Default 10.
    threads: int
        Number of threads to use.
        Use None (default) for parallel processing.
    random_seed: int
        Random seed.
        Default 1.
    """

    def __init__(self, params, random_seed=1):
        params = deepcopy(params)
        self.seasonality = params.pop('seasonality', None)
        self.early_stopping_rounds = params.pop('early_stopping_rounds', 10)

        self.df_seasonality = params.pop('df_seasonality')
        self.benchmark_model = params.pop('benchmark_model', 'y_hat_naive2')

        threads = params.pop('threads', None)
        if threads is None:
            threads = mp.cpu_count()

        #init_params = {
        #    'objective': 'multiclass',
        #    'nthread': threads,
        #    'seed': random_seed
        #}
        init_params = {
            'objective': 'multi:softprob',
            'nthread': threads,
            'seed': random_seed,
            'disable_default_eval_metric': 1
        }

        self.params = {**params, **init_params}

        self.random_seed = random_seed

    def fobj(self, predt, dtrain):
        """
        """
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape)#,
                          #order='F')
        #preds_transformed = softmax(preds, axis=1)
        preds_transformed = preds

        weighted_avg_loss_func = (preds_transformed * self.contribution_to_error[y, :]).sum(axis=1).reshape((n_train, 1))

        grad = preds_transformed * (self.contribution_to_error[y, :] - weighted_avg_loss_func)
        hess = self.contribution_to_error[y,:] * preds_transformed * (1.0 - preds_transformed) - grad * preds_transformed
        #hess = grad*(1 - 2*preds_transformed)
        return grad.flatten(), hess.flatten() #grad.flatten('F'), hess.flatten('F')

    def feval(self, predt, dtrain):
        """
        """
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape)#,
                          #order='F')
        #preds_transformed = softmax(preds, axis=1)
        preds_transformed = preds
        weighted_avg_loss_func = (preds_transformed * self.contribution_to_error[y, :]).sum(axis=1)
        fforma_loss = weighted_avg_loss_func.mean()

        return 'FFORMA-loss', fforma_loss#, False

    def calc_errors(self, preds_df, y_panel_df, y_insample_df):
        errors = calc_errors_widing(preds_df=preds_df,
                                    y_panel_df=y_panel_df,
                                    y_insample_df=y_insample_df,
                                    seasonality=self.df_seasonality,
                                    benchmark_model=self.benchmark_model)

        return errors

    def fit(self, X_train_df, preds_train_df=None,
            y_train_df=None, y_insample_df=None, verbose=True,
            errors=None, X_test_df=None, preds_test_df=None,
            y_test_df=None):
        """Fits FFORMA.


        Parameters
        ----------
        X_train_df: pandas df
            Pandas DataFrame with features.
        preds_train_df: pandas df or None
            Pandas DataFrame with columns ['unique_id', 'ds'].
            Predictions for the validation set.
            Use None to calculate on the fly.
        y_train_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y'].
        """
        # self.errors, self.features, self.val_predictions, \
        #     self.train_df, self.val_df, \
        #         self.full_df = self.pre_fit(X_df=X_df, y_df=y_df,
        #                                     val_predictions=val_predictions,
        #                                     errors=errors,
        #                                     features=features)
        features = X_train_df.set_index('unique_id').values

        print('Calculating errors...')
        if errors is None:
            errors = self.calc_errors(preds_df=preds_train_df,
                                      y_panel_df=y_train_df,
                                      y_insample_df=y_insample_df)
        else:
            errors = errors.set_index('unique_id')

        best_models_count = errors.idxmin(axis=1).value_counts()
        best_models_count = pd.Series(best_models_count, index=errors.columns)
        loser_models = best_models_count[best_models_count.isna()].index.to_list()

        if len(loser_models) > 0:
            print('Models {} never win.'.format(' '.join(loser_models)))
            print('Removing it...\n')
            errors = errors.copy().drop(columns=loser_models)

        self.contribution_to_error = errors.values
        best_models = self.contribution_to_error.argmin(axis=1)

        print('Training...')
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

        #dtrain = lgb.Dataset(data=feats_train, label=indices_train)
        #dvalid = lgb.Dataset(data=feats_val, label=indices_val)
        dtrain = xgb.DMatrix(data=feats_train, label=indices_train)
        dvalid = xgb.DMatrix(data=feats_val, label=indices_val)
        #valid_sets = [dtrain, dvalid]
        valid_sets = [(dtrain, 'train'), (dvalid, 'valid')]

        self.gbm_model_ = xgb.train(
            params=params,
            #train_set=dtrain,
            dtrain=dtrain,
            #fobj=self.fobj,
            obj=self.fobj,
            num_boost_round=num_round,
            feval=self.feval,
            #valid_sets=valid_sets,
            evals=valid_sets,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=verbose
        )

        if X_test_df is not None:
            y_hat_df = self.predict(X_test_df,
                                    base_model_preds=preds_test_df)

            y_hat_df = y_hat_df.sort_values(['unique_id', 'ds'])
            y_test_df = y_test_df.sort_values(['unique_id', 'ds'])

            self.test_min_smape = evaluate_panel(y_test=y_test_df, y_hat=y_hat_df,
                                                 y_train=None, metric=smape)['error'].mean()
            self.test_min_mape = evaluate_panel(y_test=y_test_df, y_hat=y_hat_df,
                                                y_train=None, metric=mape)['error'].mean()

        return self

    def predict(self, X_df, tmp=1, base_model_preds=None):
        """Predicts FFORMA.


        Parameters
        ----------
        X_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds'] and exogenous vars.
        tmp: float

        base_model_preds: pandas df or None
            Pandas DataFrame with columns ['unique_id', 'ds'] and
            predictions to ensemble.
            Predictions for the validation set.
            Use None to calculate on the fly.

        Return
        ------
        pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y_hat']
            where 'y_hat' denotes the fforma predictions.
        """
        check_is_fitted(self, 'gbm_model_')
        #TODO: assert match X_df and self.full_df and features

        if base_model_preds is None:
            base_model_preds = self._fit_predict_base_models(train_df=self.full_df,
                                                             val_df=X_df)
        features = X_df.set_index('unique_id')

        scores = self.gbm_model_.predict(xgb.DMatrix(features.values))#, raw_score=True)
        weights = scores #softmax(scores / tmp, axis=1)

        base_model_preds = base_model_preds.set_index(['unique_id', 'ds'])

        weights = pd.DataFrame(weights,
                               index=features.index,
                               columns=base_model_preds.columns)

        y_hat = weights * base_model_preds

        y_hat_df = base_model_preds
        y_hat_df['y_hat'] = y_hat.sum(axis=1)
        y_hat_df = y_hat_df.reset_index()

        return y_hat_df

##############################################################################
# QFFORMA
##############################################################################

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