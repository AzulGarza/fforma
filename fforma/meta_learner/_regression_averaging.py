#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from fforma.base import FQRA, QRAL1
from fforma.base.trainer import BaseModelsTrainer


class MetaLearnerFQRA(object):
    """
    Factor Quantile Regression Averaging.

    Parameters
    ----------
    tau: float
        Number in (0, 1). Quantile objective.
    n_components: int
        Number of components used to ensemble.
    scheduler: str
        Dask scheduler. Using 'processes' as default. See https://docs.dask.org/en/latest/setup/single-machine.html
        for details.
        Using "threads" can cause severe conflicts.
    """

    def __init__(self, tau: float, n_components: int, scheduler: str = 'processes'):

        self.tau = tau
        self.n_components = n_components
        self.scheduler = scheduler
        self.model = {'FQRA': FQRA(n_components=self.n_components, tau=self.tau)}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> MetaLearnerFQRA:
        """
        Fits Factor Quantile Regression Averaging.

        Parameters
        ----------
        X: pandas DataFrame.
            DataFrame with columns unique_id, ds and models to ensemble.
        """
        trainer_ = BaseModelsTrainer(models=self.model, scheduler=self.scheduler)
        trainer_.fit(X, y)

        self.trainer_ = trainer_

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        """
        check_is_fitted(self)
        y_hat = self.trainer_.predict(X)
        y_hat = y_hat[['unique_id','ds','FQRA']]
        y_hat.columns = ['unique_id', 'ds', 'y_hat']

        return y_hat

class MetaLearnerLQRA(object):
    """
    Lasso Quantile Regression Averaging.

    Parameters
    ----------
    tau: float
        Number in (0, 1). Quantile objective.
    penalty: float
        Size of penalty to perform LASSO regression.
    scheduler: str
        Dask scheduler. Using 'processes' as default. See https://docs.dask.org/en/latest/setup/single-machine.html
        for details.
        Using "threads" can cause severe conflicts.
    """

    def __init__(self, tau, penalty, scheduler='processes'):
        self.tau = tau
        self.penalty = penalty
        self.scheduler = scheduler
        self.model = {'LQRA': QRAL1(tau=self.tau, lambd=self.penalty)}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> MetaLearnerLQRA:
        """
        Fits Factor Quantile Regression Averaging.

        Parameters
        ----------
        X: pandas DataFrame.
            DataFrame with columns unique_id, ds and models to ensemble.
        """
        trainer_ = BaseModelsTrainer(models=self.model, scheduler=self.scheduler)
        trainer_.fit(X, y)

        self.trainer_ = trainer_

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        """
        check_is_fitted(self)
        y_hat = self.trainer_.predict(X)
        y_hat = y_hat[['unique_id','ds','LQRA']]
        y_hat.columns = ['unique_id', 'ds', 'y_hat']

        return y_hat
