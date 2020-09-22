#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import dask

import dask.dataframe as dd
import multiprocessing as mp
from dask.diagnostics import ProgressBar
from dask import delayed, compute
from collections import ChainMap
from functools import partial
from itertools import chain
from copy import deepcopy
from fforma.utils.reshaping import long_to_wide, train_to_horizontal, wide_to_long

from sklearn.utils.validation import check_is_fitted


class BaseModelsTrainer:
    """
    Train models to ensemble.

    Parameters
    ----------
    models: dict
        Dictionary of models to train. Ej {'ARIMA': ARIMA}
    scheduler: str
        Dask scheduler. See https://docs.dask.org/en/latest/setup/single-machine.html
        for details.
        Using "threads" can cause severe conflicts.
    """

    def __init__(self, models, scheduler='processes', partitions=None):
        self.models = models
        self.scheduler = scheduler
        self.partitions = 3 * mp.cpu_count() if partitions is None else partitions


    def fit_batch(self, batch, models):
        df_models = pd.DataFrame(index=batch.index)

        for col in models.keys():
            df_models[col] = None

        for uid, df in batch.groupby('unique_id'):
            y = df['y'].values.item()
            y = np.array(y)

            X = df['X'].values.item() if 'X' in df.columns else None

            for model_name, model in models.items():
                model = deepcopy(model)
                fitted_model = model.fit(X, y)

                df_models.loc[uid, model_name] = fitted_model

        return df_models

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'BaseModelsTrainer':
        """For each time series fit each model in models.

        Parameters
        ----------
        X: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds'] and exogenous vars.
        y: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y'].

        """
        if X is None:
            y_panel_df = long_to_wide(y)
        else:
            y_panel_df = train_to_horizontal(X, y)

        y_panel_df_dask = dd.from_pandas(y_panel_df.set_index('unique_id').sample(frac=1),
                                         npartitions=self.partitions)
        y_panel_df_dask = y_panel_df_dask.to_delayed()

        fit_batch = partial(self.fit_batch, models=self.models)

        task = [delayed(fit_batch)(part) for part in y_panel_df_dask]

        with ProgressBar():
            fitted_models = compute(*task, scheduler=self.scheduler)

        self.fitted_models_ = pd.concat(fitted_models)

        return self

    def predict_batch(self, batch, models):
        forecasts = pd.DataFrame(index=batch.index)
        for col in models.keys():
            forecasts[col] = None

        for uid, df in batch.groupby('unique_id'):
            if 'horizon' in df.columns:
                h = df['horizon'].values.item()
                df_test = range(h)
            elif 'X' in df.columns:
                df_test = df['X'].values.item()

            for model_name in models.keys():
                model = df[model_name].values.item()
                y_hat = model.predict(df_test)

                forecasts.loc[uid, model_name] = y_hat

        return forecasts

    def predict(self, X):
        """Predict each univariate model for each time series.

        X: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds']
        """
        check_is_fitted(self, 'fitted_models_')

        y_hat_df = long_to_wide(X)
        y_hat_df['horizon'] = y_hat_df['ds'].apply(lambda x: x.shape[0])


        panel_df = y_hat_df.set_index('unique_id')
        panel_df = panel_df.filter(items=['horizon', 'X']).join(self.fitted_models_)

        panel_df_dask = dd.from_pandas(panel_df, npartitions=self.partitions).to_delayed()

        predidct_batch = partial(self.predict_batch, models=self.models)

        task = [delayed(predidct_batch)(part) for part in panel_df_dask]

        with ProgressBar():
            forecasts = compute(*task, scheduler=self.scheduler)

        forecasts = pd.concat(forecasts).reset_index()

        forecasts = y_hat_df.merge(forecasts, how='left', on=['unique_id']).drop('horizon', 1)
        
        forecasts = wide_to_long(forecasts, ['ds'] + list(self.models.keys()))
        forecasts = forecasts

        return forecasts
