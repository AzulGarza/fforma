import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import multiprocessing as mp
from dask import delayed, compute
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from itertools import product, chain
import itertools

from functools import partial
from statsmodels.api import add_constant

import numpy as np
import pandas as pd

from src.l1qr import L1QR

from src.metrics.metrics import smape, mape

from src.base_models import FQRA, QRAL1
from src.meta_model import MetaModels

#############################################################################
# COMMON
#############################################################################

def evaluate_batch(batch, metric):
    losses = []
    for uid, df in batch.groupby('unique_id'):
        y = df['y'].values
        y_hat = df['y_hat'].values

        loss = metric(y, y_hat)
        losses.append(loss)
    return losses

def evaluate_panel(y_panel, y_hat_panel, metric):
    """
    """
    metric_name = metric.__code__.co_name
    y_df = y_panel.merge(y_hat_panel, how='left', on=['unique_id', 'ds'])
    y_df = y_df.set_index('unique_id')

    parts = mp.cpu_count() - 1
    y_df_dask = dd.from_pandas(y_df, npartitions=parts).to_delayed()

    evaluate_batch_p = partial(evaluate_batch, metric=metric)

    task = [delayed(evaluate_batch_p)(part) for part in y_df_dask]

    with ProgressBar():
        losses = compute(*task)

    losses = list(chain(*losses))

    mean_loss = np.mean(losses)

    return mean_loss

#############################################################################
# MEAN ENSEMBLE
#############################################################################

class MetaLearnerMean(object):
    """Evaluates ensemble model on the fly using neural networks.

    Parameters
    ----------
    actual_y: numpy array
        Actual values of the time series.
        Numpy array of size N * h
    preds_y_val: numpy array
        Model predictions to ensemble.
        Numpy array of size N * h * m.
    h: int
        Horizon of the validation set.
    weights: numpy array
        Weighted errors.
    loss_function: pytorch loss function

    random_seed:

    """
    def __init__(self, params):
        pass

    def fit(self, preds_df_test=None, y_df_test=None, verbose=True):

        y_hat_df = preds_df_test[['unique_id', 'ds']]
        y_hat_df['y_hat'] = preds_df_test.drop(['unique_id','ds'], axis=1).mean(axis=1)

        self.test_min_smape = evaluate_panel(y_panel=y_df_test,
                                             y_hat_panel=y_hat_df,
                                             metric=smape)

        self.test_min_mape = evaluate_panel(y_panel=y_df_test,
                                            y_hat_panel=y_hat_df,
                                            metric=mape)

        return self

    def predict(self, preds_df_test):
        y_hat_df = preds_df_test[['unique_id', 'ds']]
        y_hat_df['y_hat'] = preds_df_test.drop(['unique_id','ds'], axis=1).mean(axis=1)

        return y_hat_df

#############################################################################
# FQRA
#############################################################################

def long_to_horizontal(long_df):
    horizontal_df = pd.DataFrame(columns=long_df.columns)
    cols_to_parse = list(set(long_df.columns)-set(['unique_id']))
    long_df = long_df.sort_values(['unique_id','ds']).reset_index(drop=True)
    unique_ids = long_df['unique_id'].unique()
    n_series = len(unique_ids)
    max_len = long_df.groupby('unique_id')['ds'].count().max()

    dict_df = {'unique_id':unique_ids,
               'ds':list(range(1, max_len+1))} #TODO: solo enteros ####################################
    padding_dict = list(itertools.product(*list(dict_df.values())))
    padding_dict = pd.DataFrame(padding_dict, columns=list(dict_df.keys()))
    df_padded = padding_dict.merge(long_df, on=['unique_id','ds'], how='outer')
    df_padded = df_padded.sort_values(['unique_id','ds']).reset_index(drop=True)

    for col in cols_to_parse:
        values = df_padded[col].values
        values = values.reshape((n_series, max_len))
        values = values.tolist()
        horizontal_df[col] = values

    horizontal_df['unique_id'] = unique_ids
    return horizontal_df

def unpad_batch(batch, x_cols):
    batch = batch.copy()

    X_list = []
    y_list = []
    ds_list = []
    for uid, df in batch.groupby('unique_id'):
        X_i = np.vstack(df[x_cols].values[0]).T.reshape(-1, len(x_cols))
        X_i = X_i[~np.isnan(X_i).any(axis=1)]
        X_list.append(X_i)

        y_i = np.array(df['y'].values[0])
        ds_i = np.array(df['ds_x'].values[0])
        ds_i = ds_i[~np.isnan(y_i)] # Only keeps if y is not null
        ds_list.append(ds_i)

        y_i = y_i[~np.isnan(y_i)]
        y_list.append(y_i)

    batch['X'] = X_list
    batch['y'] = y_list
    batch['ds'] = ds_list

    return batch

def train_to_horizontal(X_df, y_df, x_cols=None):
    if x_cols is None:
        x_cols = list(set(X_df.columns)-set(['unique_id','ds']))

    x_horizontal = long_to_horizontal(X_df)
    y_horizontal = long_to_horizontal(y_df)
    train_df = x_horizontal.merge(y_horizontal, on='unique_id', how='outer')

    for i, row in train_df.iterrows():
        assert len(row['ds_x'])==len(row['ds_y']), 'ds_x and ds_y not corresponding'

    train_df = train_df.set_index('unique_id')

    parts = 3 * mp.cpu_count()
    train_df = dd.from_pandas(train_df.sample(frac=1), npartitions=parts).to_delayed()

    unpad_batch_p = partial(unpad_batch, x_cols=x_cols)

    task = [delayed(unpad_batch_p)(part) for part in train_df]

    with ProgressBar():
        train_df = compute(*task, scheduler='processes')

    train_df = pd.concat(train_df).reset_index()
    train_df = train_df[['unique_id','X','y','ds']]

    return train_df

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

class FactorQuantileRegressionAveraging:

    def __init__(self, tau, n_components, scheduler='processes'):
        self.tau = tau
        self.n_components = n_components
        self.scheduler = scheduler
        self.model = {'FQRA': FQRA(n_components=self.n_components, tau=self.tau)}

    def fit(self, X_df, y_df, X_test_df, y_test_df):
        """
        """
        train_df = train_to_horizontal(X_df, y_df)
        train_df['seasonality']= 12 #TODO: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXoXXXXXXXXXXXXX
        test_df = train_to_horizontal(X_test_df, y_test_df)

        self.meta_model = MetaModels(models=self.model, scheduler=self.scheduler)
        self.meta_model.fit(y_panel_df=train_df)

        y_hat_df = self.meta_model.predict(test_df)
        y_hat_df = y_hat_df[['unique_id','ds','FQRA']]
        y_hat_df.columns = ['unique_id', 'ds', 'y_hat']

        self.y_hat_df = y_hat_df

        y_hat_df = wide_to_long(y_hat_df, lst_cols=['y_hat','ds'])

        self.y_hat_df = y_hat_df

        self.test_min_smape = evaluate_panel(y_panel=y_test_df,
                                             y_hat_panel=y_hat_df,
                                             metric=smape)
        self.test_min_mape = evaluate_panel(y_panel=y_test_df,
                                            y_hat_panel=y_hat_df,
                                            metric=mape)

        return self

    def predict(self, X_df):
        """
        """

        return y_hat


#############################################################################
# LASSO QRA
#############################################################################

class LassoQuantileRegressionAveraging:

    def __init__(self, tau, penalty, scheduler='processes'):
        self.tau = tau
        self.penalty = penalty
        self.scheduler = scheduler
        self.model = {'LQRA': QRAL1(tau=self.tau, lambd=self.penalty)}

    def fit(self, X_df, y_df, X_test_df, y_test_df):
        """
        """
        train_df = train_to_horizontal(X_df, y_df)
        train_df['seasonality']= 12 #TODO: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXoXXXXXXXXXXXXX
        test_df = train_to_horizontal(X_test_df, y_test_df)

        self.meta_model = MetaModels(models=self.model, scheduler=self.scheduler)
        self.meta_model.fit(y_panel_df=train_df)

        y_hat_df = self.meta_model.predict(test_df)
        y_hat_df = y_hat_df[['unique_id','ds','LQRA']]
        y_hat_df.columns = ['unique_id', 'ds', 'y_hat']

        self.y_hat_df = y_hat_df

        y_hat_df = wide_to_long(y_hat_df, lst_cols=['y_hat','ds'])

        self.y_hat_df = y_hat_df

        self.test_min_smape = evaluate_panel(y_panel=y_test_df,
                                             y_hat_panel=y_hat_df,
                                             metric=smape)
        self.test_min_mape = evaluate_panel(y_panel=y_test_df,
                                            y_hat_panel=y_hat_df,
                                            metric=mape)

        return self

    def predict(self, X_df):
        """
        """
        y_hat = self.y_hat_df

        return y_hat
