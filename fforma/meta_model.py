#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import dask

from dask.diagnostics import ProgressBar
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
        Dictionary of models to train. Ej {'ARIMA': ARIMA}
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
            Pandas DataFrame with columns ['unique_id', 'seasonality', 'y']
        """

        fitted_models = []
        uids = []
        name_models = []
        for ts, meta_model in product(y_panel_df.groupby('unique_id'), self.models.items()):
            uid, df = ts

            y = df['y'].item()
            y = np.array(y)

            seasonality = df['seasonality'].item()

            name_model, model = deepcopy(meta_model)
            model = model(seasonality)

            fitted_model = dask.delayed(model.fit)(None, y) #TODO: correct None
            fitted_models.append(fitted_model)
            uids.append(uid)
            name_models.append(name_model)

        task = dask.delayed(fitted_models)
        with ProgressBar():
            fitted_models = task.compute(scheduler=self.scheduler)

        fitted_models = pd.DataFrame.from_dict({'unique_id': uids,
                                                'model': name_models,
                                                'fitted_model': fitted_models})

        fitted_models = fitted_models.set_index(['unique_id', 'model']).unstack()
        fitted_models = fitted_models.droplevel(0, 1).reset_index()
        fitted_models.columns.name = ''

        self.fitted_models_ = fitted_models.set_index(['unique_id'])

        return self

    def predict(self, y_hat_df):
        """Predict each univariate model for each time series.

        y_hat_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'horizon']
        """
        check_is_fitted(self, 'fitted_models_')

        y_hat_df = deepcopy(y_hat_df).set_index('unique_id')

        forecasts = []
        uids = []
        for uid, df in y_hat_df.groupby('unique_id'):
            h = df['horizon'].item()
            df_test = range(h)

            models = self.fitted_models_.loc[[uid]]
            model_forecasts = []
            for model_name in self.models.keys():
                model = models[model_name].item()
                y_hat = dask.delayed(model.predict)(df_test)
                model_forecasts.append(y_hat)

            forecasts.append(model_forecasts)
            uids.append(uid)

        task = dask.delayed(forecasts)
        with ProgressBar():
            forecasts = task.compute(scheduler=self.scheduler)

        forecasts = pd.DataFrame(forecasts,
                                 index=uids,
                                 columns=self.fitted_models_.columns)

        forecasts = forecasts.rename_axis('unique_id')

        forecasts = y_hat_df.join(forecasts).reset_index()


        return forecasts
