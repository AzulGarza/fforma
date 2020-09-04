#!/usr/bin/env python
# coding: utf-8

import torch as t
import torch.nn as nn
import multiprocessing as mp
import numpy as np
import pandas as pd

from functools import partial

# check https://github.com/ElementAI/N-BEATS/tree/master/common for other losses

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0

    return result

#############################################################################
# FORECASTING LOSSES
#############################################################################


class MAPELoss(nn.Module):
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

    Returns
    -------
    mape:
    Mean absolute percentage error.
    """
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y, y_hat):
        delta_y = t.abs((y - y_hat))
        mape = divide_no_nan(delta_y, y)
        mape = 100 * t.mean(mape)

        return mape


class SMAPE1Loss(nn.Module):
    """SMAPE1 Loss

    Calculates Symmetric Mean Absolute Percentage Error.
    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    values for the prediction and true value at a given time,
    then averages these devations over the length of the series.

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
    [1] http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf
    """
    def __init__(self):
        super(SMAPE1Loss, self).__init__()

    def forward(self, y, y_hat):
        delta_y = t.abs((y - y_hat))
        scale = y + y_hat
        smape = divide_no_nan(delta_y, scale)
        smape = 200 * t.mean(smape)

        return smape


class SMAPE2Loss(nn.Module):
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
    def __init__(self):
        super(SMAPE2Loss, self).__init__()

    def forward(self, y, y_hat):
        delta_y = t.abs((y - y_hat))
        scale = t.abs(y) + t.abs(y_hat)
        smape = divide_no_nan(delta_y, scale)
        smape = 200 * t.mean(smape)

        return smape

class MASELoss(nn.Module):
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
    def __init__(self, seasonality):
        super(MASELoss, self).__init__()
        self.seasonality = seasonality

    def forward(self, y, y_hat, y_insample):
        delta_y = t.abs((y - y_hat))
        scale = t.mean(t.abs(y_insample[:, self.seasonality:] - \
                             y_insample[:, :-self.seasonality]), axis=1)
        mase = divide_no_nan(delta_y, scale[:, None])
        mase = 100 * t.mean(mase)

        return mase


class PinballLoss(nn.Module):
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
    def __init__(self, tau=0.5):
        super(PinballLoss, self).__init__()
        self.tau = tau

    def forward(self, y, y_hat):
        delta_y = t.sub(y, y_hat)
        pinball = t.max(t.mul(self.tau, delta_y), t.mul((self.tau - 1), delta_y))
        pinball = t.mean(pinball)

        return pinball

class WeightedPinballLoss(nn.Module):
    """WeightedPinball Loss
    Computes the weighted pinball loss between y and y_hat.

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
        weighted accuracy for the predicted quantile
    """
    def __init__(self, tau=0.5):
        super(WeightedPinballLoss, self).__init__()
        self.tau = tau

    def forward(self, y, y_hat, weights):
        delta_y = t.sub(y, y_hat)
        pinball = t.max(t.mul(self.tau, delta_y), t.mul((self.tau - 1), delta_y))
        pinball = t.sum(pinball, dim=1)
        pinball = t.mul(pinball, weights)
        pinball = t.mean(pinball)
        return pinball

class ScaledWeightedPinballLoss(nn.Module):
    """WeightedPinball Loss
    Computes the weighted pinball loss between y and y_hat.

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
        scaled-weighted accuracy for the predicted quantile
    """
    def __init__(self, tau=0.5):
        super(ScaledWeightedPinballLoss, self).__init__()
        self.tau = tau

    def forward(self, y, y_hat, weights):
        scale = t.abs(y_hat) + t.abs(y)
        scale[scale == 0] = 1
        #scale = t.sum(scale, dim=1)

        delta_y = t.sub(y, y_hat)

        pinball = t.max(t.mul(self.tau, delta_y), t.mul((self.tau - 1), delta_y))
        #pinball = t.sum(pinball, dim=1)
        pinball = t.div(pinball, scale)
        pinball = t.sum(pinball, dim=1)
        pinball = t.mul(weights, pinball)

        pinball = 200 * t.mean(pinball)
        return pinball

def pinball_loss(forecast: t.Tensor, target: t.Tensor, mask: t.Tensor, tau: float) -> t.float:
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    weights = divide_no_nan(mask, t.abs(target.data) + t.abs(forecast.data))
    delta_y = t.sub(target, forecast)

    pinball = t.max(t.mul(tau, delta_y), t.mul((tau - 1), delta_y))
    pinball = pinball * weights
    pinball = t.sum(pinball) / t.sum(mask)

    return 100 * pinball

def mape_loss(target: t.Tensor, forecast: t.Tensor, mask: t.Tensor) -> t.float:
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    weights = divide_no_nan(mask, target)
    mape = t.abs((target - forecast) * weights)
    mape = t.sum(mape) / t.sum(mask)

    return 100 * mape

def smape_2_loss(target: t.Tensor, forecast: t.Tensor, mask: t.Tensor) -> t.float:
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    weights = divide_no_nan(mask, t.abs(target.data) + t.abs(forecast.data))
    smape = t.abs(target - forecast) * weights
    smape = t.sum(smape) / t.sum(mask)

    return 200 * smape
