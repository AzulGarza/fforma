#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import multiprocessing as mp

from meta_model import MetaModels
from meta_learner import MetaLearner
from meta_evaluation import calc_errors
from base_models import SeasonalNaive, Naive2, RandomWalkDrift
from tsfeatures.metrics import AVAILABLE_METRICS
from tsfeatures import tsfeatures
from sklearn.utils.validation import check_is_fitted


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

    def __init__(self, params, seasonality, dict_freqs=None, base_models=None,
                 meta_learner=MetaLearner,
                 metric='owa',
                 threads=None, random_seed=1,
                 benchmark_model='Naive2'):

        self.seasonality = seasonality
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.meta_learner = meta_learner

        self.dict_freqs = dict_freqs if dict_freqs is not None else {'D': 1, 'M': 12,
                                                                     'Q': 4, 'Y': 1,
                                                                     'Q-DEC': 4}

        if base_models is None:
            self.base_models = {'SeasonalNaive': SeasonalNaive(seasonality),
                                'Naive2': Naive2(seasonality),
                                'RandomWalkDrift': RandomWalkDrift()}
        else:
            self.base_models = base_models

        assert metric in AVAILABLE_METRICS, "Metric not specified in metrics.py"

        if threads is None:
            threads = mp.cpu_count()

        init_params = {
            'objective': 'multiclass',
            'nthread': threads,
            'seed': random_seed
        }

        self.params = {**params, **init_params}
        self.benchmark_model = benchmark_model
        self.df_season = None

    def _validation_holdout(self, X_df, y_df):
        """Splits the data in train and validation sets."""
        val = y_df.groupby('unique_id').tail(self.h)
        train = y_df.groupby('unique_id').apply(lambda df: df.head(-self.h)).reset_index(drop=True)

        return train, val

    def _fit_predict_base_models(self, train_df, val_df):
        #TODO: en el futuro revisarlo
        """Fits base models using MetaModels class."""
        meta_models = MetaModels(self.base_models, self.df_season)
        print('Fitting...')
        meta_models.fit(train_df)
        print('Predicting...')
        predictions = meta_models.predict(val_df)

        return predictions

    def _compute_base_models_errors(self, predictions, insample):
         # TODO: no funciona, solo hay 1 seasonality
        """Calculates validation errrors."""
        seasonality = 1 if self.seasonality is None else self.seasonality

        errors = calc_errors(y_panel_df=predictions,
                             y_insample_df=insample,
                             seasonality=seasonality,
                             benchmark_model=self.benchmark_model)

        return errors

    def _compute_features(self, df):
        # TODO: no funciona, solo hay 1 seasonality
        """Wrapper of tsfeatures."""
        features = tsfeatures(ts=df,
                              freq=self.seasonality,
                              dict_freqs=self.dict_freqs)

        return features

    def pre_fit(self, X_df, y_df, val_predictions=None, errors=None, features=None):
        """Calculates errors and features if needed.


        Parameters
        ----------
        X_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds'] and exogenous vars.
        y_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y'].
        val_predictions: pandas df or None
            Pandas DataFrame with columns ['unique_id', 'ds', 'y'].
            Predictions for the validation set.
            Use None to calculate on the fly.
        features: pandas df or None
            Pandas DataFrame with columns ['unique_id'] and the features.
            Use None to calculate on the fly.
        """
        #TODO: cambiar este merge
        #full_df = X_df.merge(y_df, on=['unique_id','ds'], how='inner')

        train_df, val_df = self._validation_holdout(X_df=X_df, y_df=y_df)

        print("="*29 + " Fitting Models " + "="*29)
        if val_predictions is None:
            if self.seasonality is None:
                df_season = full_df.groupby('unique_id')['ds'].apply(lambda x: pd.infer_freq(x))
                df_season = df_season.rename('freq').reset_index()
                df_season['seasonality'] = df_season['freq'].replace(self.dict_freqs)
                self.df_season = df_season
            else:
                self.df_season = None

            val_predictions = self._fit_predict_base_models(train_df=train_df,
                                                            val_df=val_df)
        print("="*28 + " Computing Errors " + "="*28)
        if errors is None:
            errors = self._compute_base_models_errors(predictions=val_predictions,
                                                      insample=train_df)

        print("="*27 + " Computing Features " + "="*27)
        if features is None:
            features = self._compute_features(df=train_df)

        return errors, features, val_predictions, train_df, val_df, y_df

    def fit(self, X_df, y_df, val_predictions=None, errors=None, features=None):
        """Fits FFORMA using MetaLearner class.


        Parameters
        ----------
        X_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds'] and exogenous vars.
        y_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y'].
        val_predictions: pandas df or None
            Pandas DataFrame with columns ['unique_id', 'ds', 'y'].
            Predictions for the validation set.
            Use None to calculate on the fly.
        features: pandas df or None
            Pandas DataFrame with columns ['unique_id'] and the features.
            Use None to calculate on the fly.
        """
        self.errors, self.features, self.val_predictions, \
            self.train_df, self.val_df, \
                self.full_df = self.pre_fit(X_df=X_df, y_df=y_df,
                                            val_predictions=val_predictions,
                                            errors=errors,
                                            features=features)

        best_models_count = self.errors.idxmin(axis=1).value_counts()
        best_models_count = pd.Series(best_models_count, index=self.errors.columns)
        loser_models = best_models_count[best_models_count.isna()].index.to_list()

        if len(loser_models) > 0:
            print('Models {} never win.'.format(' '.join(loser_models)))
            print('Removing it...\n')
            self.errors = self.errors.copy().drop(columns=loser_models)

        contribution_to_error = self.errors.values
        best_models = contribution_to_error.argmin(axis=1)

        print('Training Meta Learner...')
        meta_learner = self.meta_learner(params=self.params,
                                         y_df=self.val_df,
                                         val_predictions=self.val_predictions[self.base_models.keys()],
                                         #contribution_to_error=contribution_to_error,
                                         random_seed=self.random_seed)

        meta_learner.fit(features=self.features, best_models=best_models,
                         early_stopping_rounds=self.early_stopping_rounds,
                         verbose_eval=False)

        self.meta_learner_ = meta_learner

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
        check_is_fitted(self, 'meta_learner_')
        #TODO: assert match X_df and self.full_df and features

        if base_model_preds is None:
            base_model_preds = self._fit_predict_base_models(train_df=self.full_df,
                                                             val_df=X_df)

        weights = self.meta_learner_.predict(features=self.features, tmp=tmp)

        weights = pd.DataFrame(weights,
                               index=self.features.index,
                               columns=self.base_models.keys())

        base_model_preds = base_model_preds.set_index('unique_id')
        y_hat = weights * base_model_preds[self.base_models.keys()]

        y_hat_df = base_model_preds
        y_hat_df['y_hat'] = y_hat.sum(axis=1)
        y_hat_df = y_hat_df.reset_index()

        return y_hat_df
