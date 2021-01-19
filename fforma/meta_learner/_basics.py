#!/usr/bin/env python
# coding: utf-8

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.utils.validation import check_is_fitted


class MetaLearnerMean(object):
    """Mean ensemble."""
    def __init__(self, benchmark: Optional[str] = None):
        self.benchmark = benchmark

    def fit(self, X: pd.DataFrame, y=None) -> 'MetaLearnerMean':
        """
        Fits Mean Ensemble.

        Parameters
        ----------
        X: pandas DataFrame.
            DataFrame with columns unique_id, ds and models to ensemble.
        """
        y_hat_ = X[['unique_id', 'ds']].copy()
        cols_to_drop = ['unique_id', 'ds']
        if self.benchmark: cols_to_drop += [self.benchmark]
        y_hat_['y_hat'] = X.drop(cols_to_drop, axis=1).mean(axis=1)

        self.y_hat_ = y_hat_

        return self

    def predict(self, X: pd.DataFrame = None) -> pd.DataFrame:
        check_is_fitted(self)

        return self.y_hat_

class MetaLearnerMedian(object):
    """Median ensemble."""
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None) -> 'MetaLearnerMedian':
        """
        Fits Median Ensemble.

        Parameters
        ----------
        X: pandas DataFrame.
            DataFrame with columns unique_id, ds and models to ensemble.
        """
        y_hat_ = X[['unique_id', 'ds']]
        y_hat_['y_hat'] = X.drop(['unique_id','ds'], axis=1).median(axis=1)

        self.y_hat_ = y_hat_

        return self

    def predict(self, X: pd.DataFrame = None) -> pd.DataFrame:
        check_is_fitted(self)

        return self.y_hat_

class MetaLearnerSoftMin(object):
    """Median ensemble."""
    def __init__(self):
        pass

    def fit(self, losses: pd.DataFrame, X: pd.DataFrame, y=None) -> 'MetaLearnerSoftMin':
        """
        Fits Median Ensemble.

        Parameters
        ----------
        losses: pandas DataFrame.
            DataFrame with columns unique_id, ds and *validation losses*.
        X: pandas DataFrame.
            DataFrame with columns unique_id, ds and models to ensemble.
        """
        errors = losses.set_index('unique_id')

        weights = softmax(-errors.values, axis=1)
        weights = pd.DataFrame(weights,
                               columns=errors.columns,
                               index=errors.index)

        y_hat = X.set_index(['unique_id', 'ds']) \
                 .mul(weights) \
                 .sum(1) \
                 .rename('y_hat') \
                 .to_frame() \
                 .reset_index()

        self.y_hat_ = y_hat

        return self

    def predict(self, X: pd.DataFrame = None) -> pd.DataFrame:
        check_is_fitted(self)

        return self.y_hat_

class MetaLearnerBestModel(object):
    """Median ensemble."""
    def __init__(self):
        pass

    def fit(self, losses: pd.DataFrame, X: pd.DataFrame, y=None) -> 'MetaLearnerBestModel':
        """
        Fits Median Ensemble.

        Parameters
        ----------
        losses: pandas DataFrame.
            DataFrame with columns unique_id, ds and *validation losses*.
        X: pandas DataFrame.
            DataFrame with columns unique_id, ds and models to ensemble.
        """
        errors = losses.set_index('unique_id')

        weights = np.zeros_like(errors.values)
        weights[np.arange(errors.shape[0]), errors.values.argmin(1)] = 1

        weights = pd.DataFrame(weights,
                               columns=errors.columns,
                               index=errors.index)

        y_hat = X.set_index(['unique_id', 'ds']) \
                 .mul(weights) \
                 .sum(1) \
                 .rename('y_hat') \
                 .to_frame() \
                 .reset_index()

        self.y_hat_ = y_hat

        return self

    def predict(self, X: pd.DataFrame = None) -> pd.DataFrame:
        check_is_fitted(self)

        return self.y_hat_
