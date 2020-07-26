#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.warnings.filterwarnings('ignore')
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

from tsfeatures.metrics import evaluate_panel
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

class FactorQuantileRegressionAveraging:

    def __init__(self, tau, n_components, add_constant=True):
        self.tau = tau
        self.n_components = n_components
        self.add_constant_ = add_constant

    def _fit_quantile_ts(self, uid, X_df, y_df, n_components, add_constant_x):
        """
        X: numpy array
        y: numpy array
        """
        y = y_df['y'].values
        X = X_df.values

        pca_model = PCA(n_components=n_components).fit(X)
        X = pca_model.transform(X)
        cols = [f'factor_{f+1}' for f in range(X.shape[1])]

        if add_constant_x:
            X = add_constant(X)
            cols = ['constant'] + cols

        if X.shape[1] > 1:
            cond_number = np.linalg.cond(X)

            assert cond_number < 1e15, f'Matrix of forecasts is ill-conditioned. {uid}\n{X.shape}'

        opt_params = QuantReg(y, X).fit(self.tau).params
        opt_params = dict(zip(cols, opt_params))
        opt_params = pd.DataFrame(opt_params, index=[uid])

        pca_model = pd.DataFrame({'model': pca_model}, index=[uid])

        return opt_params, pca_model

    def _predict_quantile_ts(self, model, X_df, add_constant_x):
        """
        """
        X = X_df.values
        X = model.transform(X)
        cols = [f'factor_{f+1}' for f in range(X.shape[1])]

        if add_constant_x:
            X = add_constant(X)
            cols = ['constant'] + cols

        X = pd.DataFrame(X, columns=cols, index=X_df.index)

        return X

    def batch(self, full_df, n_components, add_constant_x):

        params, pca_models = [], []

        for uid, df in full_df.groupby('unique_id'):
            y = df[['y']]
            X = df.drop(columns=['ds', 'y'])
            param, pca_model = self._fit_quantile_ts(uid, X, y,
                                                     n_components,
                                                     add_constant_x)
            params.append(param)
            pca_models.append(pca_model)

        return params, pca_models

    def fit(self, X_df, y_df, X_test_df, y_test_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """

        #grouped_X = X_df.set_index(['unique_id', 'ds']).groupby('unique_id')
        #grouped_y = y_df.set_index(['unique_id', 'ds']).groupby('unique_id')
        full_df = X_df.merge(y_df, how='left', on=['unique_id', 'ds'])
        full_df = full_df.set_index('unique_id')

        parts = mp.cpu_count() - 1
        full_df_dask = dd.from_pandas(full_df, npartitions=parts)
        full_df_dask = full_df_dask.to_delayed()

        batch = partial(self.batch, n_components=self.n_components,
                        add_constant_x=self.add_constant_)

        task = [delayed(batch)(part) for part in full_df_dask]

        with ProgressBar():
            params_models = compute(*task, scheduler='processes')

        params, models = zip(*params_models)
        params, models = list(chain(*params)), list(chain(*models))

        self.weigths_ = pd.concat(params).fillna(0).rename_axis('unique_id')
        self.models_ = pd.concat(models)

        y_hat_df = self.predict(X_test_df)

        self.test_min_smape = evaluate_panel(y_test=y_test_df, y_hat=y_hat_df,
                                             y_train=None, metric=smape)['error'].mean()
        self.test_min_mape = evaluate_panel(y_test=y_test_df, y_hat=y_hat_df,
                                            y_train=None, metric=mape)['error'].mean()

        return self

    def predict(self, X_df):
        """
        """
        check_is_fitted(self, 'models_')
        partial_predict_quantile_ts = partial(self._predict_quantile_ts,
                                              add_constant_x=self.add_constant_)

        X_transformed = []
        for uid, X in X_df.set_index(['unique_id', 'ds']).groupby('unique_id'):
            model = self.models_.loc[uid, 'model']
            transformed = delayed(partial_predict_quantile_ts)(model, X)
            X_transformed.append(transformed)

        with ProgressBar():
            X_transformed = compute(*X_transformed)

        X_transformed = pd.concat(X_transformed).fillna(0)
        y_hat = (self.weigths_ * X_transformed).sum(axis=1)
        y_hat.name = 'y_hat'
        y_hat = y_hat.to_frame().reset_index()

        return y_hat

class LassoQuantileRegressionAveraging:

    def __init__(self, tau, penalty=1):
        self.tau = tau
        self.penalty = penalty

    def _fit_quantile_ts(self, uid, X_df, y_df, tau):
        """
        X: numpy array
        y: numpy array
        """
        y = y_df['y']
        X = X_df
        model = L1QR(y, X, tau).fit()

        model = pd.DataFrame({'model': model}, index=[uid])

        return model

    def fit(self, X_df, y_df, X_test_df, y_test_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """

        grouped_X = X_df.set_index(['unique_id', 'ds']).groupby('unique_id')
        grouped_y = y_df.set_index(['unique_id', 'ds']).groupby('unique_id')

        partial_quantile_ts = partial(self._fit_quantile_ts,
                                      tau=self.tau)

        uids = list(grouped_y.groups.keys())

        models = []
        for uid in uids:
            X, y = grouped_X.get_group(uid), grouped_y.get_group(uid)
            model = delayed(partial_quantile_ts)(uid, X, y)
            models.append(model)

        with ProgressBar():
            models = compute(*models, scheduler='processes')

        self.models_ = pd.concat(models)

        y_hat_df = self.predict(X_test_df)

        self.test_min_smape = evaluate_panel(y_test=y_test_df, y_hat=y_hat_df,
                                             y_train=None, metric=smape)['error'].mean()
        self.test_min_mape = evaluate_panel(y_test=y_test_df, y_hat=y_hat_df,
                                            y_train=None, metric=mape)['error'].mean()

        return self

    def predict(self, X_df):
        """
        """
        check_is_fitted(self, 'models_')

        y_hat = []
        for uid, X in X_df.set_index(['unique_id', 'ds']).groupby('unique_id'):
            model = self.models_.loc[uid, 'model']
            preds = delayed(model.predict)(X, self.penalty)
            y_hat.append(preds)

        with ProgressBar():
            y_hat = compute(*y_hat)

        y_hat = pd.concat(y_hat).rename('y_hat').to_frame().reset_index()
        return y_hat

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

def evaluate_model_prediction(y_train_df, outputs_df, seasonalities):
    """
    Evaluate the model against baseline Naive2 model in y_test_df
    Args:
      y_train_df: pandas df
        panel with columns unique_id, ds, y
      X_test_df: pandas df
        panel with columns unique_id, ds, x
      y_test_df: pandas df
        panel with columns unique_id, ds, y, y_hat_naive2
    """

    y_df = outputs_df.filter(['unique_id', 'ds', 'y'])
    y_hat_df = outputs_df.filter(['unique_id', 'ds', 'y_hat'])
    y_naive2_df = outputs_df.filter(['unique_id', 'ds', 'y_hat_naive2'])
    y_naive2_df.rename(columns={'y_hat_naive2': 'y_hat'}, inplace=True)
    y_insample = y_train_df.filter(['unique_id', 'ds', 'y'])

    model_owa, model_mase, model_smape = owa(y_df, y_hat_df,
                                             y_naive2_df, y_insample,
                                             seasonalities=seasonalities)

    return model_owa, model_mase, model_smape

def owa(y_panel, y_hat_panel, y_naive2_panel, y_insample, seasonalities):
    """
    Calculates MASE, sMAPE for Naive2 and current model
    then calculatess Overall Weighted Average.
    y_panel: pandas df
    panel with columns unique_id, ds, y
    y_hat_panel: pandas df
    panel with columns unique_id, ds, y_hat
    y_naive2_panel: pandas df
    panel with columns unique_id, ds, y_hat
    y_insample: pandas df
    panel with columns unique_id, ds, y (train)
    this is used in the MASE
    seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
    return: OWA
    """
    total_mase = evaluate_panel(y_panel, y_hat_panel, mase,
                                y_insample, seasonalities)
    total_mase_naive2 = evaluate_panel(y_panel, y_naive2_panel, mase,
                                        y_insample, seasonalities)
    total_smape = evaluate_panel(y_panel, y_hat_panel, smape)
    total_smape_naive2 = evaluate_panel(y_panel, y_naive2_panel, smape)

    assert len(total_mase) == len(total_mase_naive2)
    assert len(total_smape) == len(total_smape_naive2)
    assert len(total_mase) == len(total_smape)

    naive2_mase = np.mean(total_mase_naive2)
    naive2_smape = np.mean(total_smape_naive2) * 100

    model_mase = np.mean(total_mase)
    model_smape = np.mean(total_smape) * 100

    model_owa = ((model_mase/naive2_mase) + (model_smape/naive2_smape))/2

    return model_owa, model_mase, model_smape


# def evaluate_panel(y_panel, y_hat_panel, metric,
#                    y_insample=None, seasonalities=None):
#     """
#     Calculates metric for y_panel and y_hat_panel
#     y_panel: pandas df
#     panel with columns unique_id, ds, y
#     y_naive2_panel: pandas df
#     panel with columns unique_id, ds, y_hat
#     y_insample: pandas df
#     panel with columns unique_id, ds, y (train)
#     this is used in the MASE
#     seasonality: int
#     main frequency of the time series
#     Quarterly 4, Daily 7, Monthly 12
#     return: list of metric evaluations
#     """
#     metric_name = metric.__code__.co_name
#
#     y_panel = y_panel.sort_values(['unique_id', 'ds'])
#     y_hat_panel = y_hat_panel.sort_values(['unique_id', 'ds'])
#     if y_insample is not None:
#         y_insample = y_insample.sort_values(['unique_id', 'ds'])
#
#     assert len(y_panel)==len(y_hat_panel)
#     assert all(y_panel.unique_id.unique() == y_hat_panel.unique_id.unique()), "not same u_ids"
#
#     evaluation = []
#     for u_id in y_panel.unique_id.unique():
#         top_row = np.asscalar(y_panel['unique_id'].searchsorted(u_id, 'left'))
#         bottom_row = np.asscalar(y_panel['unique_id'].searchsorted(u_id, 'right'))
#         y_id = y_panel[top_row:bottom_row].y.to_numpy()
#
#         top_row = np.asscalar(y_hat_panel['unique_id'].searchsorted(u_id, 'left'))
#         bottom_row = np.asscalar(y_hat_panel['unique_id'].searchsorted(u_id, 'right'))
#         y_hat_id = y_hat_panel[top_row:bottom_row].y_hat.to_numpy()
#         assert len(y_id)==len(y_hat_id)
#
#         if metric_name == 'mase':
#             assert (y_insample is not None) and (seasonalities is not None)
#             #seasonality = seasonalities[u_id]
#             freq = u_id[0]
#             seasonality = FREQ_DICT[freq]
#
#             top_row = np.asscalar(y_insample['unique_id'].searchsorted(u_id, 'left'))
#             bottom_row = np.asscalar(y_insample['unique_id'].searchsorted(u_id, 'right'))
#             y_insample_id = y_insample[top_row:bottom_row].y.to_numpy()
#             evaluation_id = delayed(metric)(y_id, y_hat_id, y_insample_id, seasonality)
#         else:
#             evaluation_id = delayed(metric)(y_id, y_hat_id)
#         evaluation.append(evaluation_id)
#
#     with ProgressBar():
#         evaluation = compute(*evaluation)
#     return evaluation

# def smape(y, y_hat):
#     """
#     Calculates Symmetric Mean Absolute Percentage Error.
#     y: numpy array
#     actual test values
#     y_hat: numpy array
#     predicted values
#     return: sMAPE
#     """
#     y = np.reshape(y, (-1,))
#     y_hat = np.reshape(y_hat, (-1,))
#     smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
#     return smape

def mase(y, y_hat, y_train, seasonality):
    """
    Calculates Mean Absolute Scaled Error.
    y: numpy array
    actual test values
    y_hat: numpy array
    predicted values
    y_train: numpy array
    actual train values for Naive1 predictions
    seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
    return: MASE
    """
    y_hat_naive = []
    for i in range(seasonality, len(y_train)):
        y_hat_naive.append(y_train[(i - seasonality)])

    masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
    if masep==0:
        print('y_train', y_train)
        print('y_hat_naive', y_hat_naive)

    mase = np.mean(abs(y - y_hat)) / masep
    return mase
