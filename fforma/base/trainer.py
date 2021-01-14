#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List

from dask import delayed, compute
import dask.dataframe as dd
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from sklearn.utils.validation import check_is_fitted

from fforma.utils.reshaping import long_to_wide, train_to_horizontal, wide_to_long


class BaseModelsTrainer:
    """
    Train models to ensemble.

    Parameters
    ----------
    models: Dict[str, Callable]
        Dictionary of models to train. Ej {'ARIMA': ARIMA}
    scheduler: str
        Dask scheduler. See https://docs.dask.org/en/latest/setup/single-machine.html
        for details.
        Using "threads" can cause severe conflicts.
    partitions: int
        Number of partitions to be used in parallel processing.
        Default to None, number of cores minus 1.
    """

    def __init__(self, models: Dict[str, Callable],
                 scheduler: str = 'processes',
                 partitions: int = None):
        self.models = models
        self.scheduler = scheduler
        self.partitions = cpu_count() - 1 if partitions is None else partitions

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'BaseModelsTrainer':
        """For each time series fit each model in models.

        Parameters
        ----------
        X: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds'] and exogenous vars.
        y: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y'].

        """
        self.fitted_models_ = _fit(X, y, self.models,
                                   self.partitions, self.scheduler)

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict each univariate model for each time series.

        X: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds']
        """
        check_is_fitted(self, 'fitted_models_')

        forecasts = _predict(X, self.models, self.fitted_models_,
                             self.partitions, self.scheduler)

        return forecasts

def _fit(X: pd.DataFrame,
         y: pd.DataFrame,
         models: Dict[str, Callable],
         partitions: int,
         scheduler: str) -> 'BaseModelsTrainer':
    """Auxiliar function to handle parallel processing."""
    if X is None:
        y_panel_df = long_to_wide(y)
    else:
        y_panel_df = train_to_horizontal(X, y)

    y_panel_df_dask = dd.from_pandas(y_panel_df.set_index('unique_id').sample(frac=1),
                                     npartitions=partitions)
    y_panel_df_dask = y_panel_df_dask.to_delayed()

    fit_batch = partial(_fit_batch, models=models)
    task = [delayed(fit_batch)(part) for part in y_panel_df_dask]

    fitted_models = compute(*task, scheduler=scheduler)
    fitted_models = pd.concat(fitted_models)

    return fitted_models

def _fit_batch(batch: pd.DataFrame, models: Dict[str, Callable]) -> pd.DataFrame:
    df_models = pd.DataFrame(index=batch.index, columns=models.keys())

    for uid, df in batch.groupby('unique_id'):
        y = df['y'].values.item()
        y = np.array(y)

        X = df['X'].values.item() if 'X' in df.columns else None

        for model_name, model in models.items():
            model = deepcopy(model)
            fitted_model = model.fit(X, y)

            df_models.loc[uid, model_name] = fitted_model

    return df_models

def _predict(X: pd.DataFrame,
             models: Dict[str, Callable],
             fitted_models: pd.DataFrame,
             partitions: int,
             scheduler: str) -> pd.DataFrame:
    """Auxiliar function to handle parallel processing."""
    y_hat_df = long_to_wide(X)
    y_hat_df['horizon'] = y_hat_df['ds'].apply(lambda x: x.shape[0])

    panel_df = y_hat_df.set_index('unique_id')
    panel_df = panel_df.filter(items=['horizon', 'X']).join(fitted_models)

    panel_df_dask = dd.from_pandas(panel_df,
                                   npartitions=partitions)
    panel_df_dask = panel_df_dask.to_delayed()
    predict_batch = partial(_predict_batch, models=models)

    task = [delayed(predict_batch)(part) for part in panel_df_dask]

    forecasts = compute(*task, scheduler=scheduler)

    forecasts = pd.concat(forecasts)
    forecasts = forecasts.reset_index()
    forecasts = y_hat_df.merge(forecasts, how='left', on=['unique_id']).drop('horizon', 1)
    forecasts = wide_to_long(forecasts, ['ds'] + list(models.keys()))

    return forecasts

def _predict_batch(batch: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    forecasts = pd.DataFrame(index=batch.index, columns=models.keys())

    for uid, df in batch.groupby('unique_id'):
        if 'horizon' in df.columns:
            h = df['horizon'].values.item()
            df_test = range(h)
        elif 'X' in df.columns:
            df_test = df['X'].values.item()

        for model_name in models.keys():
            model = deepcopy(df[model_name].values.item())
            y_hat = model.predict(df_test)

            forecasts.loc[uid, model_name] = y_hat

    return forecasts
