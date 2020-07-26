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
from functools import partial

import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
from statsmodels.regression.quantile_regression import QuantReg
from src.l1qr import L1QR

from src.metrics.metrics import smape, mape


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


#############################################################################
# FQRA
#############################################################################

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

    def batch_train(self, batch, n_components, add_constant_x):

        params, pca_models = [], []

        for uid, df in batch.groupby('unique_id'):
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

        full_df = X_df.merge(y_df, how='left', on=['unique_id', 'ds'])
        full_df = full_df.set_index('unique_id')

        parts = mp.cpu_count() - 1
        full_df_dask = dd.from_pandas(full_df, npartitions=parts)
        full_df_dask = full_df_dask.to_delayed()

        batch_train = partial(self.batch_train, n_components=self.n_components,
                              add_constant_x=self.add_constant_)

        task = [delayed(batch_train)(part) for part in full_df_dask]

        with ProgressBar():
            params_models = compute(*task, scheduler='processes')

        params, models = zip(*params_models)
        params, models = list(chain(*params)), list(chain(*models))

        self.weigths_ = pd.concat(params).fillna(0).rename_axis('unique_id')
        self.models_ = pd.concat(models)

        y_hat_df = self.predict(X_test_df)

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


#############################################################################
# LASSO QRA
#############################################################################

class LassoQuantileRegressionAveraging:

    def __init__(self, tau, penalty=1):
        self.tau = tau
        self.penalty = penalty

    def batch_train(self, batch, tau):
        fitted_models = []

        for uid, df in batch.groupby('unique_id'):
            y = df['y']
            X = df.drop(columns=['ds', 'y'])

            model = L1QR(y, X, tau).fit()
            model = pd.DataFrame({'model': model}, index=[uid])

            fitted_models.append(model)

        return fitted_models

    def fit(self, X_df, y_df, X_test_df, y_test_df):
        """
        X: pandas df
            Panel DataFrame with columns unique_id, ds, models to ensemble
        y: pandas df
            Panel Dataframe with columns unique_id, df, y
        """
        full_df = X_df.merge(y_df, how='left', on=['unique_id', 'ds'])
        full_df = full_df.set_index('unique_id')

        parts = mp.cpu_count() - 1
        full_df_dask = dd.from_pandas(full_df, npartitions=parts)
        full_df_dask = full_df_dask.to_delayed()

        batch_train = partial(self.batch_train, tau=self.tau)

        task = [delayed(batch_train)(part) for part in full_df_dask]

        with ProgressBar():
            models = compute(*task, scheduler='processes')

        models = list(chain(*models))

        self.models_ = pd.concat(models)

        y_hat_df = self.predict(X_test_df)

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


