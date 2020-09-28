#!/usr/bin/env python
# coding: utf-8

from math import sqrt

import numpy as np

from ..utils import divide_no_nan


AVAILABLE_METRICS = ['mse', 'rmse', 'mape', 'smape', 'mase', 'rmsse',
                     'mini_owa', 'pinball_loss']


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculates Mean Squared Error.

    MSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values

    Return
    ------
    scalar: MSE
    """
    mse = np.mean(np.square(y - y_hat))

    return mse

def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculates Root Mean Squared Error.

    RMSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: RMSE
    """
    rmse = sqrt(np.mean(np.square(y - y_hat)))

    return rmse

def mape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error.

    MAPE measures the relative prediction accuracy of a
    forecasting method by calculating the percentual deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: MAPE
    """
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    mape = divide_no_nan(delta_y, scale)
    mape = np.mean(mape)
    mape = 100 * mape

    return mape

def smape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error.

    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: SMAPE
    """
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = 200 * np.mean(smape)

    assert smape <= 200, 'SMAPE should be lower than 200'

    return smape

def mase(y: np.ndarray, y_hat: np.ndarray,
         y_train: np.ndarray, seasonality: int = 1) -> float:
    """Calculates the M4 Mean Absolute Scaled Error.

    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    y_train: numpy array
      actual train values for Naive1 predictions
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1

    Return
    ------
    scalar: MASE
    """
    scale = np.mean(abs(y_train[seasonality:] - y_train[:-seasonality]))
    mase = np.mean(abs(y - y_hat)) / scale
    mase = 100 * mase

    return mase

def rmsse(y: np.ndarray, y_hat: np.ndarray,
          y_train: np.ndarray, seasonality: int = 1) -> float:
    """Calculates the M5 Root Mean Squared Scaled Error.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1

    Return
    ------
    scalar: RMSSE
    """
    scale = np.mean(np.square(y_train[seasonality:] - y_train[:-seasonality]))
    rmsse = sqrt(mse(y, y_hat) / scale)
    rmsse = 100 * rmsse

    return rmsse

def mini_owa(y: np.ndarray, y_hat: np.ndarray,
             y_train: np.ndarray,
             seasonality: int,
             y_bench: np.ndarray) -> float:
    """Calculates the Overall Weighted Average for a single series.

    MASE, sMAPE for Naive2 and current model
    then calculatess Overall Weighted Average.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array of len h (forecasting horizon)
        predicted values
    seasonality: int
        main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1
    y_train: numpy array
        insample values of the series for scale
    y_bench: numpy array of len h (forecasting horizon)
        predicted values of the benchmark model

    Return
    ------
    return: mini_OWA
    """
    mase_y = mase(y, y_hat, y_train, seasonality)
    mase_bench = mase(y, y_bench, y_train, seasonality)

    smape_y = smape(y, y_hat)
    smape_bench = smape(y, y_bench)

    mini_owa = (mase_y / mase_bench + smape_y / smape_bench) / 2

    return mini_owa

def pinball_loss(y: np.ndarray, y_hat: np.ndarray, tau: int = 0.5) -> np.ndarray:
    """Calculates the Pinball Loss.

    The Pinball loss measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for tau is 0.5 for the deviation from the median.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    tau: float
      Fixes the quantile against which the predictions are compared.

    Return
    ------
    return: pinball_loss
    """
    delta_y = y - y_hat
    pinball = np.maximum(tau * delta_y, (tau - 1) * delta_y)
    pinball = pinball.mean()

    return pinball
