#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
from itertools import product
from functools import partial
from statsmodels.api import add_constant
from statsmodels.regression.quantile_regression import QuantReg
import multiprocessing as mp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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

def plot_grid_prediction(pandas_df, columns, plot_random=True, unique_ids=None, save_file_name = None):
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

class FactorQuantilRegressionAveraging:

    def __init__(self, tau, n_components):
        self.tau = tau
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

        factor_cols = [f'factor_{f+1}' for f in range(self.n_components)]
        self.cols = ['constant'] + factor_cols

    def _fit_particual_ts(self, uid, df):
        """
        X: numpy array
        y: numpy array
        """
        forecasts = df.drop(columns=['unique_id', 'ds', 'y'])
        forecasts, _ = trimm_correlated(forecasts)
        forecasts = forecasts.values
        X = self.pca.fit_transform(forecasts)
        X = add_constant(X)

        cond_numder = np.linalg.cond(X)

        assert cond_numder < 1e15, f'Matrix of forecasts is ill-conditioned. {uid}\n{X.shape}'

        y = df[['y']].values.flatten()

        opt_params = QuantReg(y, X).fit(self.tau).params

        opt_params = dict(zip(self.cols, opt_params))
        opt_params = pd.DataFrame(opt_params, index=[uid])

        return opt_params

    def fit(self, pandas_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """

        partial_particular_ts = partial(self._fit_particual_ts)

        grouped = pandas_df.groupby('unique_id')

        # with mp.Pool() as pool:
        #     params = pool.starmap(partial_particular_ts, grouped)
        params = [partial_particular_ts(uid, df) for uid, df in grouped]

        self.weigths_ = pd.concat(params)
        self.weigths_ = self.weigths_.rename_axis('unique_id')

        return self

    def prepapre_df(self, uid, df):
        forecasts = df.drop(columns=['unique_id', 'ds', 'x'])
        forecasts, _ = trimm_correlated(forecasts)
        forecasts = forecasts.values
        X = self.pca.fit_transform(forecasts)
        X = add_constant(X)

        X = pd.DataFrame(X, columns=self.cols, index=df.set_index(['unique_id', 'ds']).index)

        return X


    def predict(self, preds):
        """
        """
        X = [self.prepapre_df(uid, df) for uid, df in preds.groupby(['unique_id'])]
        X = pd.concat(X)

        y_hat = (self.weigths_ * X).sum(axis=1)
        y_hat.name = 'y_hat'
        y_hat = y_hat.to_frame()
        y_hat = y_hat.reset_index()

        return y_hat


class QuantilRegressionAveraging:

    def __init__(self, tau):
        self.tau = tau

    def _fit_particual_ts(self, uid, df):
        """
        X: numpy array
        y: numpy array
        """
        X = df.drop(columns=['unique_id', 'ds', 'y'])
        col_models = X.columns
        X, cols_interest = trimm_correlated(X)
        X = X.values
        X = add_constant(X)

        col_models = np.concatenate((['constant'], col_models))
        cols_interest = np.concatenate(([True], cols_interest))

        cond_numder = np.linalg.cond(X)

        assert cond_numder < 1e15, f'Matrix of forecasts is ill-conditioned. {uid}\n{X.shape}'

        y = df[['y']].values.flatten()

        opt_params = np.zeros(col_models.shape)
        params = QuantReg(y, X).fit(self.tau).params

        opt_params[cols_interest] = params

        opt_params = dict(zip(col_models, opt_params))

        opt_params = pd.DataFrame(opt_params, index=[uid])

        return opt_params

    def fit(self, pandas_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """

        partial_particular_ts = partial(self._fit_particual_ts)

        grouped = pandas_df.groupby('unique_id')

        # with mp.Pool() as pool:
        #     params = pool.starmap(partial_particular_ts, grouped)
        params = [partial_particular_ts(uid, df) for uid, df in grouped]

        self.weigths_ = pd.concat(params)
        self.weigths_ = self.weigths_.rename_axis('unique_id')

        return self

    def predict(self, preds):
        """
        """
        preds = preds.copy()
        preds['constant'] = 1
        preds = preds.set_index(['unique_id', 'ds'])
        y_hat = (self.weigths_ * preds).sum(axis=1)
        y_hat.name = 'y_hat'
        y_hat = y_hat.to_frame()
        y_hat = y_hat.reset_index()

        return y_hat

def trimm_correlated(df_in, threshold=0.99):
    """

    Notes
    -----
    [1] Based on https://stackoverflow.com/questions/49282049/remove-strongly-correlated-columns-from-dataframe
    """
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_corr[df_corr.isna()] = 1
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
    constant_cols = (df_in[df_not_correlated.index] != df_in[df_not_correlated.index].iloc[0]).any()

    df_not_correlated = df_not_correlated & constant_cols

    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]

    return df_out, df_not_correlated.values
