#!/usr/bin/env python
# coding: utf-8

from itertools import count
from numbers import Number
from sys import float_info
from typing import List, Optional


import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.stattools import adfuller


def embed(x: np.array, p: int) -> np.array:
    """Embeds the time series x into a low-dimensional Euclidean space.

    Parameters
    ----------
    x: numpy array
        Time series.
    p: int
        Embedding dimension.

    Notes
    -----
    [1] embed(x, p) = embed(x, [0, 1, ..., p - 1])

    References
    ----------
    [1] https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/embed
    """
    is_p_int = isinstance(p, int)

    if is_p_int and p == 0:
        raise Exception('Embedding dimension should not be 0')

    rolls = range(p) if is_p_int else p
    min_p = p - 1 if is_p_int else np.max(p)

    x = np.transpose(np.vstack(list((np.roll(x, k) for k in rolls))))
    x = x[min_p:]

    return x

class QuantileAutoRegression:
    """
    Perform Quantile Regression on a time series using lags.
        y_t = c + a_1 y_{t - n1} + a_2 + y_{t - n2} + ...
    Where n1, n2, ... are indexes provided by the user.
    A Dicky-Fuller test is performed to decide if the process
    is stationary. If not, the time series is differentiated
    as many times as needed (max number of differences can be
    controlled by the user).

    Parameters
    ----------
    tau: float
        Quantile to predict between (0, 1).
    ar_terms: list[int]
        List of autorregresive terms to add.
    add_constant: bool
        Wheter add + c to the model.
    max_diffs: int
        Max number of differences to apply.

    Notes
    -----
    [1] To avoid Dicky-Fuller test just use max_diffs = 0.
    [2] Be cautious when the time series is too short.
    [3] Setting tau = 0.5 equals to optimize for MAE.

    Examples
    --------
    For 90 percentile (over-estimate) for daily data:
        model = QuantileAutoRegression(0.9, ar_terms=[7, 14])
    """

    def __init__(self, tau: float,
                 ar_terms: List[int],
                 add_constant: bool = True,
                 max_diffs: int = 10):
        self.tau = tau
        self.ar_terms = ar_terms
        self.add_constant = add_constant
        self.max_diffs = max_diffs

        self.min_ar, self.max_ar = np.min(ar_terms), np.max(ar_terms)

        self.differences: int
        self.is_constant: bool
        self.last_y_train: Number
        self.y_train: np.ndarray
        self.model_: RegressionResultsWrapper

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileAutoRegression':
        y = y.copy()
        self.last_y_train = y[-1]
        self.is_constant = np.var(y) == 0

        # Convert y to an stationary process
        self.differences = 0
        for _ in range(self.max_diffs):
            _, pval, *_ = adfuller(y)
            if pval < 0.05:
                break
            y = np.diff(y, 1)
            self.differences += 1

        design_mat = embed(y, [0] + self.ar_terms)
        self.y_train, X_train = design_mat[:, 0], design_mat[:, 1:]

        if self.add_constant:
            X_train = np.hstack([X_train, np.ones((len(X_train), 1))])

        if np.linalg.cond(X_train) > 1 / float_info.epsilon and not self.is_constant:
            raise Exception('X matrix is ill-conditioned '
                            'try reducing number of ar_terms '
                            'or setting add_constant=False.')

        self.model_ = QuantReg(self.y_train, X_train).fit(self.tau)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        horizon = len(X)

        y_hat = self.y_train
        len_train = self.y_train.size
        forecast_size = len_train + horizon

        while y_hat.size < forecast_size:
            y_hat_placeholder = np.zeros(self.min_ar)
            y_hat = np.concatenate([y_hat, y_hat_placeholder])

            y_test = embed(y_hat, self.ar_terms)[-self.min_ar:]

            if self.add_constant:
                y_test = np.hstack([y_test, np.ones((len(y_test), 1))])

            y_hat[-self.min_ar:] = self.model_.predict(y_test)

        y_hat = y_hat[len_train:forecast_size]

        if self.differences:
            for i in range(self.differences): y_hat = y_hat.cumsum()
            y_hat += self.last_y_train

        return y_hat
