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
from fforma.l1qr import L1QR
from ESRNN.m4_data import prepare_m4_data


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

    def __init__(self, tau, n_components, add_constant=True):
        self.tau = tau
        self.n_components = n_components
        self.add_constant = add_constant

    def _fit_quantile_ts(self, uid):
        """
        X: numpy array
        y: numpy array
        """
        y = self.grouped_y.get_group(uid)['y'].values
        X = self.grouped_X.get_group(uid).values

        pca_model = PCA(n_components=self.n_components).fit(X)
        X = pca_model.transform(X)
        cols = [f'factor_{f+1}' for f in range(X.shape[1])]

        if add_constant:
            X = add_constant(X)
            cols = ['constant'] + cols

        if X.shape[1] > 1:
            cond_number = np.linalg.cond(X)

            assert cond_number < 1e15, f'Matrix of forecasts is ill-conditioned. {uid}\n{X.shape}'

        opt_params_l = []

        for tau in self.tau:
            opt_params = QuantReg(y, X).fit(tau).params

            opt_params = dict(zip(cols, opt_params))
            tau = 100 * tau
            tau = int(tau)
            tau = 'p' + str(tau)
            index = pd.MultiIndex.from_arrays([[uid], [tau]], names=('unique_id', 'quantile'))
            opt_params = pd.DataFrame(opt_params, index=index)

            opt_params_l.append(opt_params)

        opt_params = pd.concat(opt_params_l)
        pca_model = pd.DataFrame({'model': pca_model}, index=[uid])

        return opt_params, pca_model

    def _predict_quantile_ts(self, uid, X_df):
        X = X_df.values
        X = self.models_.loc[uid, 'model'].transform(X)
        cols = [f'factor_{f+1}' for f in range(X.shape[1])]

        if self.add_constant:
            X = add_constant(X)
            cols = ['constant'] + cols

        X = pd.DataFrame(X, columns=cols, index=X_df.index)

        return X

    def fit(self, X_df, y_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """

        self.grouped_X = X_df.groupby('unique_id')
        self.grouped_y = y_df.groupby('unique_id')
        self.uids = list(self.grouped_y.groups.keys())

        partial_quantile_ts = partial(self._fit_quantile_ts)

        with mp.Pool() as pool:
           params_models = pool.map(partial_quantile_ts, self.uids)

        params, models = zip(*params_models)

        self.weigths_ = pd.concat(params).fillna(0)
        self.models_ = pd.concat(models)

        return self

    def predict(self, X_df):
        """
        """
        partial_predict_quantile_ts = partial(self._predict_quantile_ts)

        with mp.Pool() as pool:
           X_transformed = pool.starmap(partial_predict_quantile_ts, X_df.groupby('unique_id'))

        X_transformed = pd.concat(X_transformed).fillna(0)

        y_hat = (self.weigths_ * X_transformed).sum(axis=1)
        y_hat.name = 'y_hat'
        y_hat = y_hat.to_frame()

        y_hat = y_hat.pivot_table(index=['unique_id','ds'], columns='quantile')
        y_hat.columns = y_hat.columns.droplevel().rename(None)

        return y_hat

class LassoQuantileRegressionAveraging:

    def __init__(self, tau, penalty=1):
        self.tau = tau
        self.penalty = penalty

    def _fit_quantile_ts(self, uid):
        """
        X: numpy array
        y: numpy array
        """
        y = self.grouped_y.get_group(uid)['y']
        X = self.grouped_X.get_group(uid)

        model = L1QR(y, X, self.tau).fit()

        return uid, model

    def _predict_quantile_ts(self, uid, X_df):

        prediction = self.models_[uid].predict(X_df, self.penalty)

        return prediction

    def fit(self, X_df, y_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """

        partial_quantile_ts = partial(self._fit_quantile_ts)

        self.grouped_X = X_df.groupby('unique_id')
        self.grouped_y = y_df.groupby('unique_id')
        self.uids = list(self.grouped_y.groups.keys())

        with mp.Pool() as pool:
            models = pool.map(partial_quantile_ts, self.uids)

        self.models_ = {uid: model for uid, model in models}

        return self

    def predict(self, X_df):
        """
        """
        check_is_fitted(self, 'models_')
        partial_predict_quantile_ts = partial(self._predict_quantile_ts)

        with mp.Pool() as pool:
            y_hat = pool.starmap(partial_predict_quantile_ts, X_df.groupby('unique_id'))

        y_hat = pd.concat(y_hat).rename(f'p{int(100 * self.tau)}').to_frame()

        return y_hat

def evaluate_forecasts(dataset_name, panel_df, directory, num_obs):
    _, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name=dataset_name,
                                                          directory=directory,
                                                          num_obs=num_obs)

    y_test = panel_df[panel_df['unique_id'].isin(y_test_df['unique_id'].unique())]

    seasonality = freqs[dataset_name]

    eval_cols = set(panel_df.columns) - {'unique_id', 'ds'}

    evaluation = {}

    for col in eval_cols:
        print(col)
        y_test_model = y_test[['unique_id', 'ds', col]].rename(columns={col: 'y_hat'})
        owa, mase, smape = evaluate_prediction_owa(y_test_model, y_train_df, X_test_df, y_test_df, seasonality)
        evaluation[col] = owa


    evaluation = pd.DataFrame(evaluation, index=[dataset_name])

    return evaluation
