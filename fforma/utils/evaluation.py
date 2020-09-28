#!/usr/bin/env python
# coding: utf-8

from functools import partial
from typing import Callable, Optional

from dask import delayed, compute
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import multiprocessing as mp
import pandas as pd

from .reshaping import long_to_wide, wide_to_long

def _evaluate_batch(batch, metric, metric_name, seasonality):
    df_losses = pd.DataFrame(index=batch.index.unique())
    df_losses[metric_name] = None

    for uid, df in batch.groupby('unique_id'):
        y = df['y'].values.item()
        y_hat = df['y_hat'].values.item()

        if metric_name in ['mase']:
            y_train = df['y_train'].values.item()
            loss = metric(y, y_hat, y_train, seasonality)
        else:
            loss = metric(y, y_hat)
        df_losses.loc[uid, metric_name] = loss

    return df_losses

def _set_y_hat(df, col, drop_y=True):
    df = df.rename(columns={col: 'y_hat'})

    if drop_y:
        df = df.drop('y', 1)

    return df

def evaluate_panel(y_panel: pd.DataFrame,
                   y_hat_panel: pd.DataFrame,
                   metric: Callable,
                   y_train_df: Optional[pd.DataFrame] = None,
                   seasonality: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluates time series panel according to metric.

    Parameters
    ----------
    y_panel: pd.DataFrame
        Pandas Data Frame with columns ['unique_id', 'ds', 'y'].
    y_hat_panel: pd.DataFrame
        Pandas Data Frame with columns ['unique_id', 'ds', 'y_hat']
    metric: Callable
        Function to calculate metric.
    y_train_df: pd.DataFrame
        Optional for particular metrics.
        Pandas Data Frame with columns ['unique_id', 'ds', 'y']
    seasonality: int
        Optional for particular metrics.
        Integer.
    """
    metric_name = metric.__name__
    y_df = y_panel.merge(y_hat_panel, how='left', on=['unique_id', 'ds'])
    y_df = long_to_wide(y_df)

    if y_train_df is not None:
        wide_y_train_df = long_to_wide(y_train_df).rename(columns={'y': 'y_train'})
        y_df = y_df.merge(wide_y_train_df, how='left', on=['unique_id'])

    y_df = y_df.set_index('unique_id')
    parts = mp.cpu_count() - 1
    y_df_dask = dd.from_pandas(y_df, npartitions=parts).to_delayed()

    evaluate_batch_p = partial(_evaluate_batch, metric=metric,
                               metric_name=metric_name,
                               seasonality=seasonality)

    task = [delayed(evaluate_batch_p)(part) for part in y_df_dask]

    with ProgressBar():
        losses = compute(*task)

    losses = pd.concat(losses).reset_index()
    losses[metric_name] = losses[metric_name].astype(float)

    return losses

def evaluate_models(y_panel: pd.DataFrame,
                    models_panel: pd.DataFrame,
                    metric: Callable,
                    y_train_df: Optional[pd.DataFrame] = None,
                    seasonality: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluates panel of models according to metric.

    Parameters
    ----------
    y_panel: pd.DataFrame
        Pandas Data Frame with columns ['unique_id', 'ds', 'y'].
    models_panel: pd.DataFrame
        Pandas Data Frame with columns ['unique_id', 'ds'] and models columns.
    metric: Callable
        Function to calculate metric.
    y_train_df: pd.DataFrame
        Optional for particular metrics.
        Pandas Data Frame with columns ['unique_id', 'ds', 'y']
    seasonality: int
        Optional for particular metrics.
        Integer.
    """
    models = set(models_panel.columns) - {'unique_id', 'ds'}
    metric_name = metric.__name__

    list_losses = []
    for model in models:
        y_hat_df = _set_y_hat(models_panel, model, False)
        loss = evaluate_panel(y_test_df, y_hat_df, metric, y_train_df, seasonality)
        loss = loss.rename(columns={metric_name: model})
        loss = loss.set_index('unique_id')
        list_losses.append(loss)

    df_losses = pd.concat(list_losses, 1).reset_index()

    return df_losses
