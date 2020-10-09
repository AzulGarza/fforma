#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import multiprocessing as mp
import pandas as pd
from scipy.special import softmax
from sklearn.utils.validation import check_is_fitted
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaLearnerXGBoost:
    """Feature-based Forecast Model Averaging (FFORMA).


    Parameters
    ----------
    xgb_params: Dict
        Parameters to be used by xgboost.
    """

    def __init__(self, xgb_params: Dict,
                 benchmark: str,
                 n_estimators: int,
                 random_seed: Optional[int] = None,
                 threads: Optional[int] = None) -> 'MetaLearnerXGBoost':
        self.threads = threads
        if self.threads is None:
            self.threads = mp.cpu_count()

        self.num_round = n_estimators
        self.random_seed = random_seed
        self.benchmark = benchmark

        init_params = {
            'objective': 'multi:softprob',
            'nthread': self.threads,
            'seed': self.random_seed,
            'disable_default_eval_metric': 1
        }

        self.params = {**xgb_params, **init_params}

        self.models = None
        self.contribution_to_error = None

    def fobj(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        # predt in softmax
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape)
        weighted_avg_loss_func = (preds * self.contribution_to_error[y, :]).sum(axis=1).reshape((n_train, 1))

        grad = preds * (self.contribution_to_error[y, :] - weighted_avg_loss_func)
        hess = self.contribution_to_error[y,:] * preds * (1.0 - preds) - grad * preds

        return grad.flatten(), hess.flatten()

    def feval(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        """
        """
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape)

        weighted_avg_loss_func = (preds * self.contribution_to_error[y, :]).sum(axis=1)
        fforma_loss = weighted_avg_loss_func.mean()

        return 'FFORMA-loss', fforma_loss

    def fit(self, features: pd.DataFrame,
            errors: pd.DataFrame) -> 'MetaLearnerXGBoost':
        """Fits FFORMA.

        Parameters
        ----------
        """
        if self.benchmark not in errors.columns:
            raise Exception(f'Benchmark {self.benchmark} must be part of errors')

        errors = errors.copy()

        self.models = errors.columns.difference(['unique_id', self.benchmark])
        self.models = list(self.models)
        for col in self.models:
            errors[col] /= errors[self.benchmark]
        errors = errors.set_index('unique_id')[self.models]

        best_models_count = errors.idxmin(axis=1).value_counts()
        best_models_count = pd.Series(best_models_count, index=errors.columns)
        loser_models = best_models_count[best_models_count.isna()].index.to_list()

        if len(loser_models) > 0:
            loser = ', '.join(loser_models)
            logger.info(f'Models {loser} never win.')
            logger.info('Removing it...\n')
            errors = errors.drop(columns=loser_models)
            for model in loser_models:
                self.models.remove(model)

        self.contribution_to_error = errors.values
        best_models = self.contribution_to_error.argmin(axis=1)

        params = deepcopy(self.params)
        params['num_class'] = len(np.unique(best_models))

        features = features.set_index('unique_id').values
        dtrain = xgb.DMatrix(data=features, label=np.arange(features.shape[0]))

        self.gbm_model_ = xgb.train(
            params=params,
            dtrain=dtrain,
            obj=self.fobj,
            num_boost_round=self.num_round,
            feval=self.feval
        )

        return self

    def predict(self, features: pd.DataFrame,
                forecasts: pd.DataFrame) -> 'MetaLearnerXGBoost':
        """Predicts FFORMA.

        Parameters
        ----------

        Returns
        -------
        """
        check_is_fitted(self, 'gbm_model_')

        features = features.set_index('unique_id')
        forecasts = forecasts.set_index(['unique_id', 'ds'])[self.models]

        weights = self.gbm_model_.predict(xgb.DMatrix(features.values))
        weights = pd.DataFrame(weights,
                               index=features.index,
                               columns=forecasts.columns)

        y_hat = weights * forecasts

        y_hat_df = pd.DataFrame(index=forecasts.index, columns=['y_hat'])
        y_hat_df['y_hat'] = y_hat.sum(axis=1)
        y_hat_df = y_hat_df.reset_index()

        return y_hat_df
