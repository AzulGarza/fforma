#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from copy import deepcopy
from tsfeatures.metrics import smape, mase, evaluate_panel

###############################################################################
########## UTILS FOR FFORMA FLOW
###############################################################################

def calc_errors(y_panel_df, y_insample_df, seasonality, benchmark_model='Naive2'):
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

    y_panel = y_panel_df[['unique_id', 'ds', 'y']]
    y_hat_panel_fun = lambda model_name: y_panel_df[['unique_id','ds', model_name]].rename(columns={model_name: 'y_hat'})

    errors_smape = y_panel_df[['unique_id']].drop_duplicates().reset_index(drop=True)
    errors_mase = errors_smape.copy()

    model_names = set(y_panel_df.columns) - set(y_panel.columns)

    y_panel = y_panel
    y_insample_df = y_insample_df

    for model_name in model_names:
        errors_smape[model_name] = None
        errors_mase[model_name] = None
        y_hat_panel = y_hat_panel_fun(model_name)

        errors_smape[model_name] = evaluate_panel(y_test=y_panel, y_hat=y_hat_panel,
                                                  y_train=y_insample_df,
                                                  metric=smape,
                                                  seasonality=seasonality)['error']
        errors_mase[model_name] = evaluate_panel(y_test=y_panel, y_hat=y_hat_panel,
                                                 y_train=y_insample_df,
                                                 metric=mase,
                                                 seasonality=seasonality)['error']

    mean_smape_benchmark = errors_smape[benchmark_model].mean()
    mean_mase_benchmark = errors_mase[benchmark_model].mean()

    errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
    errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

    errors = errors_smape/mean_mase_benchmark + errors_mase/mean_smape_benchmark
    errors = 0.5*errors

    return errors

def get_prediction_panel(y_panel_df, h, freq):
    """Constructs panel to use with
    predict method.
    """
    df = y_complete_train_df[['unique_id', 'ds']].groupby('unique_id').max().reset_index()

    predict_panel = []
    for idx, df in df.groupby('unique_id'):
        date = df['ds'].values.item()
        unique_id = df['unique_id'].values.item()

        date_range = pd.date_range(date, periods=4, freq='D')
        df_ds = pd.DataFrame.from_dict({'ds': date_range})
        df_ds['unique_id'] = unique_id
        predict_panel.append(df_ds[['unique_id', 'ds']])

    predict_panel = pd.concat(predict_panel)

    return predict_panel
