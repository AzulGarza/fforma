#!/usr/bin/env python
# coding: utf-8

import torch as t
import torch.nn as nn

from ..utils import divide_no_nan


def mape_loss(y, y_hat, mask=None):
    """MAPE Loss

    Calculates Mean Absolute Percentage Error between
    y and y_hat. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the
    percentual deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    mape:
    Mean absolute percentage error.
    """
    weights = divide_no_nan(mask, y)
    return t.mean(t.abs((y - y_hat) * t.abs(weights)))

def MSEloss(y, y_hat, mask=None):
    """MSE Loss

    Calculates Mean Squared Error between
    y and y_hat. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the
    percentual deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    mse:
    Mean Squared Error.
    """
    mse = (y - y_hat)**2
    mse = mask * mse
    mse = t.mean(mse)
    return mse


def SMAPELoss(y, y_hat, mask=None):
    """SMAPE2 Loss

    Calculates Symmetric Mean Absolute Percentage Error.
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
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.

    Returns
    -------
    smape:
        symmetric mean absolute percentage error

    References
    ----------
    [1] https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)
    """
    if mask is None:
        mask = t.ones(y_hat.size())
    delta_y = t.abs(y - y_hat)
    scale = t.abs(y) + t.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = smape * mask
    smape = 200 * t.mean(smape)

    return smape


def MASELoss(y, y_hat, y_insample, seasonality, mask=None) :
    """ Calculates the M4 Mean Absolute Scaled Error.

    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.

    Parameters
    ----------
    seasonality: int
        main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1
    y: tensor (batch_size, output_size)
        actual test values
    y_hat: tensor (batch_size, output_size)
        predicted values
    y_train: tensor (batch_size, input_size)
        actual insample values for Seasonal Naive predictions

    Returns
    -------
    mase:
        mean absolute scaled error

    References
    ----------
    [1] https://robjhyndman.com/papers/mase.pdf
    """
    if mask is None:
        mask = t.ones(y_hat.size())
    delta_y = t.abs(y - y_hat)
    scale = t.mean(t.abs(y_insample[:, seasonality:] - \
                            y_insample[:, :-seasonality]), axis=1)
    mase = divide_no_nan(delta_y, scale[:, None])
    mase = mase * mask
    mase = t.mean(mase)
    return mase


def pinball_loss(y, y_hat, mask=None, tau=0.5):
    """Pinball Loss
    Computes the pinball loss between y and y_hat.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    tau: float, between 0 and 1
        the slope of the pinball loss, in the context of
        quantile regression, the value of tau determines the
        conditional quantile level.

    Returns
    -------
    pinball:
        average accuracy for the predicted quantile
    """
    if mask is None:
        mask = t.ones(y_hat.size())
    delta_y = t.sub(y, y_hat)
    pinball = t.max(t.mul(tau, delta_y), t.mul((tau - 1), delta_y))
    pinball = pinball * mask
    pinball = t.mean(pinball)

    return pinball

def smape_1_loss(forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
    """
    sMAPE loss as defined in "Appendix A" of
    http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    return 200 * t.mean(divide_no_nan(t.abs(forecast - target), forecast.data + target.data) * mask)
