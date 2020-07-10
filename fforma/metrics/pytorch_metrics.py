#!/usr/bin/env python
# coding: utf-8

import torch as t
import torch.nn as nn
import multiprocessing as mp
import pandas as pd

from functools import partial

# check https://github.com/ElementAI/N-BEATS/tree/master/common for other losses

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == float('inf')] = .0

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

    def forward(self, y, y_hat):
        delta_y = t.sub(y, y_hat)
        pinball = t.max(t.mul(self.tau, delta_y), t.mul((self.tau - 1), delta_y))
        pinball = 2 * t.sum(pinball) / t.sum(t.abs(y))

        return pinball

######################################################################
# PANEL EVALUATION
######################################################################

def _evaluate_ts_torch(uid, y_test, y_hat,
                       y_train, metric,
                       seasonality, y_bench, metric_name):
    y_test_uid = y_test.loc[uid].y.values
    y_hat_uid = y_hat.loc[uid].y_hat.values

    y_test_uid = t.tensor(y_test_uid, dtype=t.float)
    y_hat_uid = t.tensor(y_hat_uid, dtype=t.float)

    if metric_name in ['mase', 'rmsse']:
        y_train_uid = y_train.loc[uid].y.values
        y_train_uid = t.tensor(y_train_uid, dtype=t.float)

        evaluation_uid = metric(y=y_test_uid, y_hat=y_hat_uid,
                                y_train=y_train_uid,
                                seasonality=seasonality)

    elif metric_name in ['mini_owa']:
        y_train_uid = y_train.loc[uid].y.values
        y_bench_uid = y_bench.loc[uid].y_hat.values

        y_train_uid = t.tensor(y_train_uid, dtype=t.float)
        y_bench_uid = t.tensor(y_bench_uid, dtype=t.float)

        evaluation_uid = metric(y=y_test_uid, y_hat=y_hat_uid,
                                y_train=y_train_uid,
                                seasonality=seasonality,
                                y_bench=y_bench_uid)

    else:
         evaluation_uid = metric(y=y_test_uid, y_hat=y_hat_uid)

    return uid, evaluation_uid.item()

def evaluate_panel_torch(y_test, y_hat, y_train,
                         metric, seasonality=None,
                         y_bench=None,
                         threads=None):
    """Calculates a specific metric written in pytorch for y and y_hat
       (and y_train, if needed).

    Parameters
    ----------
    y_test: pandas df
        df with columns ['unique_id', 'ds', 'y']
    y_hat: pandas df
        df with columns ['unique_id', 'ds', 'y_hat']
    y_train: pandas df
        df with columns ['unique_id', 'ds', 'y'] (train)
        This is used in the scaled metrics ('mase', 'rmsse').
    seasonality: int
        Main frequency of the time series.
        Used in ('mase', 'rmsse').
        Commonly used seasonalities:
            Hourly: 24,
            Daily: 7,
            Weekly: 52,
            Monthly: 12,
            Quarterly: 4,
            Yearly: 1.
    y_bench: pandas df
        df with columns ['unique_id', 'ds', 'y_hat']
        predicted values of the benchmark model
        This is used in 'mini_owa'.
    threads: int
        Number of threads to use. Use None (default) for parallel processing.

    Return
    ------
    list of metric evaluations for each unique_id
        in the panel data
    """
    metric_name = metric.__class__.__name__
    uids = y_test['unique_id'].unique()
    y_hat_uids = y_hat['unique_id'].unique()

    assert len(y_test)==len(y_hat), "not same length"
    assert all(uids == y_hat_uids), "not same u_ids"

    y_test = y_test.set_index(['unique_id', 'ds'])
    y_hat = y_hat.set_index(['unique_id', 'ds'])

    if metric_name in ['mase', 'rmsse']:
        y_train = y_train.set_index(['unique_id', 'ds'])
    elif metric_name in ['mini_owa']:
        y_train = y_train.set_index(['unique_id', 'ds'])
        y_bench = y_bench.set_index(['unique_id', 'ds'])

    partial_evaluation = partial(_evaluate_ts_torch,
                                 y_test=y_test, y_hat=y_hat,
                                 y_train=y_train, metric=metric,
                                 seasonality=seasonality,
                                 y_bench=y_bench,
                                 metric_name=metric_name)

    with mp.Pool(threads) as pool:
        evaluations = pool.map(partial_evaluation, uids)

    evaluations = pd.DataFrame(evaluations, columns=['unique_id', 'error'])

    return evaluations
