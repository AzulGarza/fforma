#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import multiprocessing as mp

from src.meta_model import MetaModels
from src.meta_learner import MetaLearner
from src.meta_evaluation import calc_errors
from src.base_models import SeasonalNaive, Naive2, RandomWalkDrift
from tsfeatures.metrics import AVAILABLE_METRICS
from tsfeatures import tsfeatures
from sklearn.utils.validation import check_is_fitted

DICT_FREQS = {'H':24, 'D': 7, 'W':52, 'M': 12, 'Q': 4, 'Y': 1}

class FFORMA(object):
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

    def __init__(self,
                 meta_learner_params,
                 meta_learner=MetaLearner,
                 metric='owa',
                 random_seed=1):

        self.meta_learner = meta_learner
        self.meta_learner_params = meta_learner_params

        #assert metric in AVAILABLE_METRICS, "Metric not specified in metrics.py"
        self.random_seed = random_seed

    def _validation_holdout(self, X_df, y_df):
        """Splits the data in train and validation sets."""
        pass

    def _fit_predict_base_models(self, train_df, val_df):
        """Fits base models using MetaModels class."""
        pass

    def _compute_base_models_errors(self, predictions, insample):
        """Calculates validation errrors."""
        pass

    def _compute_features(self, df):
        """Wrapper of tsfeatures."""
        pass

    def pre_fit(self, X_df, y_df, val_predictions=None, errors=None, features=None):
        """Calculates errors and features if needed."""
        pass

    def fit(self, X_train_df, preds_train_df, y_train_df, verbose=True):
        """
        Fit
        """
        self.meta_learner = self.meta_learner(self.meta_learner_params)
        self.meta_learner.fit(X_train_df, preds_train_df, y_train_df, verbose=verbose)
        self._fitted = True

        return self
        

    def predict(self, X_test_df, preds_test_df, y_df):
        """
        """
        y_df = y_df[['unique_id', 'ds', 'y', 'y_hat_naive2']]
        y_df = y_df.sort_values(['unique_id','ds']).reset_index(drop=True)

        y_df['y_hat'] = self.meta_learner.predict(X_test_df, preds_test_df)

        return y_df
