#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.warnings.filterwarnings('ignore')
import warnings
warnings.warn = lambda *a, **kw: False
import pandas as pd
import random
from itertools import product, chain
from functools import partial
from statsmodels.api import add_constant
from statsmodels.regression.quantile_regression import QuantReg
import multiprocessing as mp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from src.l1qr import L1QR
from ESRNN.m4_data import prepare_m4_data
from ESRNN.utils_evaluation import evaluate_prediction_owa
from dask import delayed, compute
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import time
from src.metrics.metrics import smape, mape

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

freqs = {'Hourly': 24, 'Daily': 1,
         'Monthly': 12, 'Quarterly': 4,
         'Weekly':1, 'Yearly': 1}

freqs_hyndman = {'H': 24, 'D': 1,
                 'M': 12, 'Q': 4,
                 'W':1, 'Y': 1}

FREQ_DICT = {'H':24, 'D': 7, 'W':52, 'M': 12, 'Q': 4, 'Y': 1, 'O': 1}

def prepare_data(df, h, min_size_per_series=None, max_series=None, regressors=None):
    """Splits the data in train and validation sets."""
    if min_size_per_series is None:
        min_size_per_series = 3 * h

    sizes = df.groupby('unique_id').size()
    uids_min_size = sizes[sizes > min_size_per_series].index.to_list()
    valid_df = df[df['unique_id'].isin(uids_min_size)]

    if max_series is not None:
        uids_max_series = valid_df.groupby(['unique_id']).sum()
        uids_max_series = uids_max_series.nlargest(max_series, 'y').index.to_list()
        valid_df = valid_df[valid_df['unique_id'].isin(uids_max_series)]

    test = valid_df.groupby('unique_id').tail(h)
    train = valid_df.groupby('unique_id').apply(lambda df: df.head(-h)).reset_index(drop=True)

    if regressors is None:
        regressors = []

    x_cols, y_cols = (['unique_id', 'ds'] + regressors), ['unique_id', 'ds', 'y']

    X_train_df = train.filter(items=x_cols)
    y_train_df = train.filter(items=y_cols)
    X_test_df = test.filter(items=x_cols)
    y_test_df = test.filter(items=y_cols)

    return X_train_df, y_train_df, X_test_df, y_test_df

def plot_grid_prediction(pandas_df, columns, plot_random=True, unique_ids=None, save_file_name=None):
    """
    y: pandas df
        panel with columns unique_id, ds, y
    y_hat: pandas df
        panel with columns unique_id, ds, y_hat
    plot_random: bool
    if unique_ids will be sampled
    unique_ids: list
    unique_ids to plot
    save_file_name: str
    file name to save plot
    """
    pd.plotting.register_matplotlib_converters()

    fig, axes = plt.subplots(2, 4, figsize = (24,8))

    if not unique_ids:
        unique_ids = pandas_df['unique_id'].unique()

    assert len(unique_ids) >= 8, "Must provide at least 8 ts"

    if plot_random:
        unique_ids = random.sample(set(unique_ids), k=8)

    for i, (idx, idy) in enumerate(product(range(2), range(4))):
        y_uid = pandas_df[pandas_df.unique_id == unique_ids[i]]

        for col in columns:
            axes[idx, idy].plot(y_uid.ds, y_uid[col], label = col)
        axes[idx, idy].set_title(unique_ids[i])
        axes[idx, idy].legend(loc='upper left')
        axes[idx, idy].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.show()

def wide_to_long(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res

def long_to_wide_uid(uid, df, full_columns, cols_to_parse):
    horizontal_df = pd.DataFrame(columns=full_columns)
    for col in cols_to_parse:
        horizontal_df[col] = [df[col].values]
    horizontal_df['unique_id'] = uid

    return horizontal_df

def long_to_wide(long_df, threads=None):

    if threads is None:
        threads = mp.cpu_count()

    cols_to_parse = set(long_df.columns) - {'unique_id'}
    partial_long_to_wide_uid = partial(long_to_wide_uid,
                                       full_columns=long_df.columns,
                                       cols_to_parse=cols_to_parse)

    with mp.Pool(threads) as pool:
        wide_df = pool.starmap(partial_long_to_wide_uid, long_df.groupby('unique_id'))

    wide_df = pd.concat(wide_df).reset_index(drop=True)

    return wide_df

def evaluate_batch(batch, metric, metric_name):
    df_losses = pd.DataFrame(index=batch.index.unique())
    df_losses[metric_name] = None

    for uid, df in batch.groupby('unique_id'):
        y = df['y'].values
        y_hat = df['y_hat'].values

        loss = metric(y, y_hat)
        df_losses.loc[uid, metric_name] = loss

    return df_losses

def evaluate_panel(y_panel, y_hat_panel, metric, y_train_df=None):
    """
    """
    metric_name = metric.__code__.co_name
    y_df = y_panel.merge(y_hat_panel, how='left', on=['unique_id', 'ds'])
    y_df = y_df.set_index('unique_id')

    parts = mp.cpu_count() - 1
    y_df_dask = dd.from_pandas(y_df, npartitions=parts).to_delayed()

    evaluate_batch_p = partial(evaluate_batch, metric=metric,
                               metric_name=metric_name)

    task = [delayed(evaluate_batch_p)(part) for part in y_df_dask]

    with ProgressBar():
        losses = compute(*task)

    losses = pd.concat(losses).reset_index()
    losses[metric_name] = losses[metric_name].astype(float)

    return losses

def set_y_hat(df, col, drop_y=True):
    df = df.rename(columns={col: 'y_hat'})

    if drop_y:
        df = df.drop('y', 1)

    return df

def evaluate_models(y_test_df, models_panel, metric, y_train_df=None):
    models = set(models_panel.columns) - {'unique_id', 'ds'}
    metric_name = metric.__name__

    list_losses = []
    for model in models:
        y_hat_df = set_y_hat(models_panel, model, False)
        loss = evaluate_panel(y_test_df, y_hat_df, metric, y_train_df)
        loss = loss.rename(columns={metric_name: model})
        loss = loss.set_index('unique_id')
        list_losses.append(loss)

    df_losses = pd.concat(list_losses, 1).reset_index()

    return df_losses
