#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.utils.validation import check_is_fitted


class MetaLearnerMean(object):
    """Mean ensemble."""
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None) -> 'MetaLearnerMean':
        """
        Fits Mean Ensemble.

        Parameters
        ----------
        X: pandas DataFrame.
            DataFrame with columns unique_id, ds and models to ensemble.
        """
        y_hat_ = X[['unique_id', 'ds']]
        y_hat_['y_hat'] = X.drop(['unique_id','ds'], axis=1).mean(axis=1)

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
