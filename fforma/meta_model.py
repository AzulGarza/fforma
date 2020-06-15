#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import dask

from collections import ChainMap
from functools import partial
from itertools import product
from copy import deepcopy

from sklearn.utils.validation import check_is_fitted


class MetaModels:
    """
    Train models to ensemble.

    Parameters
    ----------
    models: dict
        Dictionary of models to train. Ej {'ARIMA': ARIMA()}
    scheduler: str
        Dask scheduler. See https://docs.dask.org/en/latest/setup/single-machine.html
        for details.
        Using "threads" can cause severe conflicts.
    """

    def __init__(self, models, scheduler='processes'):
        self.models = models
        self.scheduler = scheduler

    def fit(self, y_panel_df):
        """For each time series fit each model in models.

        y_panel_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        """

        fitted_models = []
        uids = []
        name_models = []
        for ts, meta_model in product(y_panel_df.groupby('unique_id'), self.models.items()):
            uid, y = ts
            y = y['y'].values
            name_model, model = deepcopy(meta_model)
            fitted_model = dask.delayed(model.fit)(None, y) #TODO: correct None
            fitted_models.append(fitted_model)
            uids.append(uid)
            name_models.append(name_model)

        fitted_models = dask.delayed(fitted_models).compute(scheduler=self.scheduler)

        fitted_models = pd.DataFrame.from_dict({'unique_id': uids,
                                                'model': name_models,
                                                'fitted_model': fitted_models})

        self.fitted_models_ = fitted_models.set_index(['unique_id', 'model'])

        return self

    def predict(self, y_hat_df):
        """Predict each univariate model for each time series.

        y_hat_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds']
        """
        check_is_fitted(self, 'fitted_models_')

        y_hat_df = deepcopy(y_hat_df)#[['unique_id', 'ds']])

        forecasts = []
        uids = []
        dss = []
        name_models = []
        for ts, name_model in product(y_hat_df.groupby('unique_id'), self.models.keys()):
            uid, df = ts
            h = len(df)
            model = self.fitted_models_.loc[(uid, name_model)]
            model = model.item()
            y_hat = dask.delayed(model.predict)(h)
            forecasts.append(y_hat)
            uids.append(np.repeat(uid, h))
            dss.append(df['ds'])
            name_models.append(np.repeat(name_model, h))

        forecasts = dask.delayed(forecasts).compute(scheduler=self.scheduler)
        forecasts = zip(uids, dss, name_models, forecasts)

        forecasts_df = []
        for uid, ds, name_model, forecast in forecasts:
            dict_df = {'unique_id': uid,
                       'ds': ds,
                       'model': name_model,
                       'forecast': forecast}
            df = dask.delayed(pd.DataFrame.from_dict)(dict_df)
            forecasts_df.append(df)

        forecasts = dask.delayed(forecasts_df).compute()
        forecasts = pd.concat(forecasts)

        forecasts = forecasts.set_index(['unique_id', 'ds', 'model']).unstack()
        forecasts = forecasts.droplevel(0, 1).reset_index()
        forecasts.columns.name = ''

        forecasts = forecasts.merge(y_hat_df, how='left',
                                    on=['unique_id', 'ds'])


        return forecasts
