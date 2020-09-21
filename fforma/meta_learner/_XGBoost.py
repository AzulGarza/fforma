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
