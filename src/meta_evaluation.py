#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from copy import deepcopy
from src.utils import long_to_wide, wide_to_long
from tsfeatures.metrics import smape, mase, evaluate_panel

###############################################################################
########## UTILS FOR FFORMA FLOW
###############################################################################

def calc_errors(preds_df, y_panel_df, y_insample_df, seasonality, benchmark_model):
    """Calculates OWA of each time series
    usign benchmark_model as benchmark.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    y_insample_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        Train set.
    seasonality: int
        Frequency of the time seires.
    benchmark_model: str
        Column name of the benchmark model.

    Returns
    -------
    Pandas DataFrame
        OWA errors for each time series and each model.
    """

    assert benchmark_model in y_panel_df.columns

    y_hat_panel_fun = lambda model_name: preds_df[['unique_id','ds', model_name]].rename(columns={model_name: 'y_hat'})

    errors_smape = y_panel_df[['unique_id']].drop_duplicates().reset_index(drop=True)
    errors_mase = errors_smape.copy()

    model_names = set(preds_df.columns) - {'unique_id', 'ds'}

    for model_name in model_names:
        errors_smape[model_name] = None
        errors_mase[model_name] = None
        y_hat_panel = y_hat_panel_fun(model_name)

        errors_smape[model_name] = evaluate_panel(y_test=y_panel_df, y_hat=y_hat_panel,
                                                  y_train=y_insample_df,
                                                  metric=smape,
                                                  seasonality=seasonality)['error']
        errors_mase[model_name] = evaluate_panel(y_test=y_panel_df, y_hat=y_hat_panel,
                                                 y_train=y_insample_df,
                                                 metric=mase,
                                                 seasonality=seasonality)['error']

    mean_smape_benchmark = errors_smape[benchmark_model].mean()
    mean_mase_benchmark = errors_mase[benchmark_model].mean()

    errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
    errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

    errors = errors_mase / mean_mase_benchmark + errors_smape / mean_smape_benchmark
    errors = 0.5 * errors

    return errors

def calc_errors_widing(preds_df, y_panel_df, y_insample_df, seasonality, benchmark_model):
    """Calculates OWA of each time series
    usign benchmark_model as benchmark.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    y_insample_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        Train set.
    seasonality: pandas df
        Frequency of the time seires.
    benchmark_model: str
        Column name of the benchmark model.

    Returns
    -------
    Pandas DataFrame
        OWA errors for each time series and each model.
    """
    df_wide = [long_to_wide(df).set_index(['unique_id']) \
               for df in (preds_df, y_panel_df.rename(columns={'y': 'y_test'}),
                          y_insample_df.rename(columns={'y': 'y_train'}))]

    df_wide = pd.concat(df_wide, axis=1).join(seasonality.set_index('unique_id'))

    assert benchmark_model in df_wide.columns

    df_wide_fun = lambda model_name: df_wide.rename(columns={model_name: 'y_hat'})

    errors_smape = y_panel_df[['unique_id']].drop_duplicates().reset_index(drop=True)
    errors_mase = errors_smape.copy()

    model_names = set(preds_df.columns) - {'unique_id', 'ds'}
    model_names.add(benchmark_model)

    for model_name in model_names:
        errors_smape[model_name] = None
        errors_mase[model_name] = None
        df_wide_hat = df_wide_fun(model_name)

        errors_smape[model_name] = evaluate_wide_panel(wide_panel=df_wide_hat,
                                                       metric=smape)['smape']
        errors_mase[model_name] = evaluate_wide_panel(wide_panel=df_wide_hat,
                                                      metric=mase)['mase']

    mean_smape_benchmark = errors_smape[benchmark_model].mean()
    mean_mase_benchmark = errors_mase[benchmark_model].mean()

    errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
    errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

    errors = errors_smape / mean_smape_benchmark + errors_mase / mean_mase_benchmark
    errors = 0.5 * errors

    return errors

def evaluate_wide_panel(wide_panel, metric, remove_na=True):

    metric_name = metric.__name__

    losses = []
    uids = []
    for uid, df in wide_panel.groupby('unique_id'):
        y_test = df['y_test'].item()
        y_test = np.array(y_test)

        y_hat = df['y_hat'].item()
        y_hat = np.array(y_hat)

        if remove_na:
            y_test = y_test[~np.isnan(y_test)]
            y_hat = y_hat[~np.isnan(y_hat)]

        if metric_name in ['mase', 'rmsse']:
            y_train = df['y_train'].item()
            y_train = np.array(y_train)

            if remove_na:
                y_train = y_train[~np.isnan(y_train)]

            seasonality = df['seasonality'].item()

            loss = metric(y=y_test, y_hat=y_hat,
                          y_train=y_train,
                          seasonality=seasonality)
        else:
            loss = metric(y_test, y_hat)

        losses.append(loss)
        uids.append(uid)

    loss_panel = [uids, losses]
    loss_panel = zip(*loss_panel)

    loss_panel = pd.DataFrame(loss_panel,
                              columns=['unique_id', metric_name])

    return loss_panel


def get_prediction_panel(y_panel_df, h, freq):
    """Constructs panel to use with
    predict method.
    """
    df = y_complete_train_df[['unique_id', 'ds']].groupby('unique_id').max().reset_index()

    predict_panel = []
    for idx, df in df.groupby('unique_id'):
        date = df['ds'].values.item()
        unique_id = df['unique_id'].values.item()

        date_range = pd.date_range(date, periods=h, freq=freq)
        df_ds = pd.DataFrame.from_dict({'ds': date_range})
        df_ds['unique_id'] = unique_id
        predict_panel.append(df_ds[['unique_id', 'ds']])

    predict_panel = pd.concat(predict_panel)

    return predict_panel
