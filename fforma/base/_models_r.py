#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import rpy2.robjects as robjects

from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.vectors import IntVector, FloatVector

forecast = importr('forecast')
stats = importr('stats')


def forecast_object_to_dict(forecast_object):
    """Transforms forecast_object into a python dictionary."""
    dict_ = zip(forecast_object.names,
                list(forecast_object))
    dict_ = dict(dict_)

    return dict_

def get_forecast(fitted_model, h):
    """Calculates forecast from a fitted model."""
    y_hat = forecast.forecast(fitted_model, h=h)
    y_hat = forecast_object_to_dict(y_hat)
    y_hat = np.array(y_hat['mean'])

    return y_hat

def fit_forecast_model(y, freq, model, **kwargs):
    """Wrapper of the following flow:
        - Load _forecast_ package.
        - Transform data into a ts object.
        - Fit the model.

    Parameters
    ----------
    model: str
        Name of a model included in the
        _forecast_ package. Ej. 'auto.arima'.
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    kwargs:
        Arguments of the model function.

    Returns
    -------
    rpy2 object
        Fitted model
    """
    pandas2ri.activate()

    freq = deepcopy(freq)

    rstring = """
     function(y, freq, ...){
         suppressMessages(library(forecast))
         y_ts <- msts(y, seasonal.periods=freq)
         fitted_model<-%s(y_ts, ...)
         fitted_model
     }
    """ % (model)

    rfunc = robjects.r(rstring)

    fitted = rfunc(FloatVector(y), freq, **kwargs)

    return fitted

class ForecastModel(BaseEstimator, RegressorMixin):
    """Wrapper for models in the R package _forecast_ that returns a model.

    Parameters
    ----------
    model: str
        Name of a model included in the
        _forecast_ package. Ej. 'auto.arima'.
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    kwargs:
        Arguments of the model function.
    """

    def __init__(self, model, freq, **kwargs):
        self.freq = freq
        self.model = model
        self.kwargs = kwargs

    def fit(self, X, y):
        self.fitted_model_ = fit_forecast_model(y, self.freq, self.model, **self.kwargs)

        return self

    def predict(self, X):
        check_is_fitted(self, 'fitted_model_')

        h = len(X)
        y_hat = get_forecast(self.fitted_model_, h)

        return y_hat

class ForecastObject(BaseEstimator, RegressorMixin):
    """Wrapper for models in the R package _forecast_ that returns an object.

    Parameters
    ----------
    model: str
        Name of a model that returns objects
        and is included in the
        _forecast_ package. Ej. 'rwf'.
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    kwargs:
        Arguments of the model function.
    """

    def __init__(self, model, freq, **kwargs):
        self.freq = freq
        self.model = model
        self.kwargs = kwargs

    def fit(self, X, y):
        self.y_ts_ = y

        return self

    def predict(self, X):
        check_is_fitted(self, 'y_ts_')

        h = len(X)

        fitted_model = fit_forecast_model(self.y_ts_, self.freq, self.model, h=h, **self.kwargs)
        y_hat = get_forecast(fitted_model, h)

        return y_hat

class ARIMA(ForecastModel):
    """Wrapper of forecast::auto.arima from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='auto.arima', freq=freq, **kwargs)

class ETS(ForecastModel):
    """Wrapper of forecast::ets from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='ets', freq=freq, **kwargs)

class NNETAR(ForecastModel):
    """Wrapper of forecast::nnetar from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='nnetar', freq=freq, **kwargs)

class TBATS(ForecastModel):
    """Wrapper of forecast::tbats from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)

    Notes
    -----
        - Disabling parallel mode by default.
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='tbats', freq=freq, **{'use.parallel': False}, **kwargs)

class STLM(ForecastModel):
    """Wrapper of forecast::stlm from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        assert freq > 1, "STLM cannot handle non seasonal time series"
        super().__init__(model='stlm', freq=freq, **kwargs)

class STLMFFORMA(ForecastModel):
    """Wrapper of
        stlm_ar_forec <- function(x, h) {
          model <- tryCatch({
            forecast::stlm(x, modelfunction = stats::ar)
          }, error = function(e) forecast::auto.arima(x, d=0,D=0))
          forecast::forecast(model, h=h)$mean
        }

        from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)

    References
    ----------
    https://github.com/robjhyndman/M4metalearning/blob/master/R/forec_methods_list.R
    """

    def __init__(self, freq, **kwargs):
        if freq > 1:
            super().__init__(model='stlm', freq=freq, modelfunction=stats.ar, **kwargs)
        else:
            super().__init__(model='auto.arima', freq=freq, d=0, D=0)

    def fit(self, X, y):
        try:
            self.fitted_model_ = fit_forecast_model(y, self.freq, self.model, **self.kwargs)
        except:
            self.fitted_model_ = fit_forecast_model(y, self.freq, 'auto.arima', d=0, D=0)

        return self

##############################################################################
######## FORECAST OBJECTS ####################################################
#############################################################################

class RandomWalk(ForecastObject):
    """Wrapper of forecast::rwf from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='rwf', freq=freq, **kwargs)

class ThetaF(ForecastObject):
    """Wrapper of forecast::thetaf from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='thetaf', freq=freq, **kwargs)

class NaiveR(ForecastObject):
    """Wrapper of forecast::naive from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='naive', freq=freq, **kwargs)

class SeasonalNaiveR(ForecastObject):
    """Wrapper of forecast::snaive from R.

    Parameters
    ----------
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities. (Last seasonality
        considered as frequency.)
    """

    def __init__(self, freq, **kwargs):
        super().__init__(model='snaive', freq=freq, **kwargs)
