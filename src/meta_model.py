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

from sklearn.utils.validation import check_is_fitted


class MetaModels:
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

    def __init__(self, models, scheduler='processes'):
        self.models = models
        self.scheduler = scheduler

    def fit_batch(self, batch, models):
        df_models = pd.DataFrame(index=batch.index)

        for col in models.keys():
            df_models[col] = None

        for uid, df in batch.groupby('unique_id'):
            y = df['y'].item()
            y = np.array(y)
            seasonality = df['seasonality'].item()

            X = df['X'].item() if 'X' in df.columns else None

            for model_name, model in models.items():
                model = deepcopy(model)
                #fitted_model = model(seasonality).fit(X, y)
                fitted_model = model.fit(X, y)

                df_models.loc[uid, model_name] = fitted_model

        return df_models

    def fit(self, y_panel_df):
        """For each time series fit each model in models.

        y_panel_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'seasonality', 'y']
        """
        parts = 3 * mp.cpu_count()
        y_panel_df_dask = dd.from_pandas(y_panel_df.set_index('unique_id').sample(frac=1),
                                         npartitions=parts)
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
                h = df['horizon'].item()
                df_test = range(h)
            elif 'X' in df.columns:
                df_test = df['X'].item()

            for model_name in models.keys():
                model = df[model_name].item()
                y_hat = model.predict(df_test)

                forecasts.loc[uid, model_name] = y_hat

        return forecasts

    def predict(self, y_hat_df):
        """Predict each univariate model for each time series.

        y_hat_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'horizon']
        """
        check_is_fitted(self, 'fitted_models_')

        panel_df = y_hat_df.set_index('unique_id')
        panel_df = panel_df.filter(items=['horizon', 'X']).join(self.fitted_models_)

        parts = 1 * mp.cpu_count()
        panel_df_dask = dd.from_pandas(panel_df, npartitions=parts).to_delayed()

        predidct_batch = partial(self.predict_batch, models=self.models)

        task = [delayed(predidct_batch)(part) for part in panel_df_dask]

        with ProgressBar():
            forecasts = compute(*task, scheduler=self.scheduler)

        forecasts = pd.concat(forecasts).reset_index()

        forecasts = y_hat_df.merge(forecasts, how='left', on=['unique_id'])

        return forecasts
