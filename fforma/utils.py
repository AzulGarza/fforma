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
from sklearn.utils.validation import check_is_fitted


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

class FactorQuantileRegressionAveraging:

    def __init__(self, tau, n_components):
        self.tau = tau
        self.n_components = n_components

        factor_cols = [f'factor_{f+1}' for f in range(self.n_components)]
        self.cols = ['constant'] + factor_cols

    def _fit_pca_ts(self, uid, X):
        forecasts, cols = trimm_correlated(X)
        forecasts = forecasts.values
        pca_model = PCA(n_components=self.n_components).fit(forecasts) if forecasts.size > 0 else None
        X_transformed = pca_model.transform(forecasts)
        X_transformed = add_constant(X_transformed)

        return uid, cols, pca_model, X_transformed

    def fit_pca(self, X_df):

        partial_fit_pca_ts = partial(self._fit_pca_ts)

        with mp.Pool() as pool:
            fitted_pca = pool.starmap(partial_fit_pca_ts, X_df.groupby('unique_id'))

        self.fitted_pca = {uid: vals for uid, *vals in fitted_pca}
        #self.fitted_pca = {uid: self._fit_pca_ts(X) for uid, X in X_df.groupby('unique_id')}

        return self

    def _fit_quantile_ts(self, uid, y_df):
        """
        X: numpy array
        y: numpy array
        """
        check_is_fitted(self, 'fitted_pca')

        _, _, X = self.fitted_pca[uid]
        y = y_df['y'].values

        if X.shape[1] > 1:
            cond_number = np.linalg.cond(X)

            assert cond_number < 1e15, f'Matrix of forecasts is ill-conditioned. {uid}\n{X.shape}'

        opt_params_l = []

        for tau in self.tau:
            opt_params = QuantReg(y, X).fit(tau).params

            opt_params = dict(zip(self.cols, opt_params))
            tau = 100 * tau
            tau = int(tau)
            tau = 'p' + str(tau)
            index = pd.MultiIndex.from_arrays([[uid], [tau]], names=('unique_id', 'quantile'))
            opt_params = pd.DataFrame(opt_params, index=index)

            opt_params_l.append(opt_params)

        opt_params = pd.concat(opt_params_l)

        return opt_params

    def _predict_quantile_ts(self, uid, X_df):
        cols, model, _  = self.fitted_pca[uid]
        X = X_df[cols].values
        X_transformed = model.transform(X)
        X_transformed = add_constant(X_transformed)
        X_transformed = pd.DataFrame(X_transformed,
                                     columns=self.cols[:X_transformed.shape[1]],
                                     index=X_df.set_index(['unique_id', 'ds']).index)

        return X_transformed

    def fit(self, X_df, y_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """

        self.fit_pca(X_df)

        partial_quantile_ts = partial(self._fit_quantile_ts)

        grouped = y_df.groupby('unique_id')

        with mp.Pool() as pool:
            params = pool.starmap(partial_quantile_ts, grouped)
        #params = [partial_quantile_ts(uid, y['y'].values) for uid, y in grouped]

        self.weigths_ = pd.concat(params)

        return self

    def predict(self, X_df):
        """
        """
        partial_predict_quantile_ts = partial(self._predict_quantile_ts)

        with mp.Pool() as pool:
            X_transformed = pool.starmap(partial_predict_quantile_ts, X_df.groupby(['unique_id']))

        X_transformed = pd.concat(X_transformed)

        y_hat = (self.weigths_ * X_transformed).sum(axis=1)
        y_hat.name = 'y_hat'
        y_hat = y_hat.to_frame()

        y_hat = y_hat.pivot_table(index=['unique_id','ds'], columns='quantile')
        y_hat.columns = y_hat.columns.droplevel().rename(None)

        return y_hat

class QuantileRegressionAveraging:

    def __init__(self, tau):
        self.tau = tau

    def _fit_trimm_correlated_ts(self, uid, X):
        forecasts, cols = trimm_correlated(X)
        cols = cols.to_list()
        forecasts = forecasts.values
        X_transformed = add_constant(forecasts)

        return uid, cols, X_transformed

    def fit_trimm_correlated(self, X_df):

        partial_fit_trimm_correlated_ts = partial(self._fit_trimm_correlated_ts)

        with mp.Pool() as pool:
            fitted_pca = pool.starmap(partial_fit_trimm_correlated_ts, X_df.groupby('unique_id'))

        self.fitted_trimm_correlated = {uid: vals for uid, *vals in fitted_pca}
        #self.fitted_pca = {uid: self._fit_pca_ts(X) for uid, X in X_df.groupby('unique_id')}

        return self

    def _fit_quantile_ts(self, uid, y_df):
        """
        X: numpy array
        y: numpy array
        """
        check_is_fitted(self, 'fitted_trimm_correlated')

        cols, X = self.fitted_trimm_correlated[uid]
        cols = ['constant'] + cols
        y = y_df['y'].values

        if X.shape[1] > 1:
            cond_number = np.linalg.cond(X)

            assert cond_number < 1e15, f'Matrix of forecasts is ill-conditioned. {uid}\n{X.shape}'

        opt_params_l = []

        for tau in self.tau:
            opt_params = QuantReg(y, X).fit(tau).params
            assert opt_params.size == len(cols)

            opt_params = dict(zip(cols, opt_params))
            tau = 100 * tau
            tau = int(tau)
            tau = 'p' + str(tau)
            index = pd.MultiIndex.from_arrays([[uid], [tau]], names=('unique_id', 'quantile'))
            opt_params = pd.DataFrame(opt_params, index=index)

            opt_params_l.append(opt_params)

        opt_params = pd.concat(opt_params_l)

        return opt_params

    def _predict_quantile_ts(self, uid, X_df):
        cols, _  = self.fitted_trimm_correlated[uid]
        forecasts = X_df[cols].values
        X_transformed = add_constant(forecasts)
        X_transformed = pd.DataFrame(X_transformed,
                                     columns=['constant'] + cols,
                                     index=X_df.set_index(['unique_id', 'ds']).index)

        return X_transformed

    def fit(self, X_df, y_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """

        self.fit_trimm_correlated(X_df)

        partial_quantile_ts = partial(self._fit_quantile_ts)

        grouped = y_df.groupby('unique_id')

        with mp.Pool() as pool:
            params = pool.starmap(partial_quantile_ts, grouped)
        #params = [partial_quantile_ts(uid, y['y'].values) for uid, y in grouped]

        self.weigths_ = pd.concat(params).fillna(0)

        return self

    def predict(self, X_df):
        """
        """
        partial_predict_quantile_ts = partial(self._predict_quantile_ts)

        with mp.Pool() as pool:
            X_transformed = pool.starmap(partial_predict_quantile_ts, X_df.groupby(['unique_id']))

        X_transformed = pd.concat(X_transformed)

        y_hat = (self.weigths_ * X_transformed).sum(axis=1)
        y_hat.name = 'y_hat'
        y_hat = y_hat.to_frame()

        y_hat = y_hat.pivot_table(index=['unique_id','ds'], columns='quantile')
        y_hat.columns = y_hat.columns.droplevel().rename(None)

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
    non_constant_cols = df_in[df_not_correlated.index].std() > 1e-8

    df_not_correlated = df_not_correlated & non_constant_cols

    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index]].index
    df_out = df_in[un_corr_idx]

    return df_out, un_corr_idx
