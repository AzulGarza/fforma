#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import logging
from time import sleep
from typing import Dict, Union, Iterable, Optional

import pandas as pd
from tsfeatures.tsfeatures_r import tsfeatures_r

from fforma.base.trainer import BaseModelsTrainer
from fforma.base import (Naive2, ARIMA, ETS, NNETAR, STLM, TBATS, STLMFFORMA,
                         RandomWalk, ThetaF, NaiveR, SeasonalNaiveR)
from fforma.experiments.datasets.tourism import TourismInfo, Tourism
from fforma.metrics.numpy import mape, smape
from fforma.utils.evaluation import evaluate_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaseData:
    features: pd.DataFrame
    forecasts: pd.DataFrame
    ground_truth: pd.DataFrame
    mape_forecasts: pd.DataFrame
    smape_forecasts: pd.DataFrame
    groups: Dict

    def get_ids(self, ids: Iterable) -> 'BaseData':
        """Return filtered data based on ids.

        Parameters
        ----------
        ids: Iterable.
            Iterable of ids.
        """
        features = self.features.query('unique_id in @ids')
        forecasts = self.forecasts.query('unique_id in @ids')
        ground_truth = self.ground_truth.query('unique_id in @ids')
        mape_forecasts = self.mape_forecasts.query('unique_id in @ids')
        smape_forecasts = self.smape_forecasts.query('unique_id in @ids')

        return BaseData(features=features, forecasts=forecasts, \
                        ground_truth=ground_truth,
                        mape_forecasts=mape_forecasts, \
                        smape_forecasts=smape_forecasts, \
                        groups={'group': ids})


    def get_group(self, group: str) -> 'BaseData':
        """Filters group data.

        Parameters
        ----------
        group: str
            Group name.
        """
        assert group in self.groups, \
            f'Please provide a valid group: {", ".join(self.groups.keys())}'

        ids = self.groups[group]

        return self.get_ids(ids)

    def get_metric(self, metric: str) -> pd.DataFrame:
        """Return metric data.

        Parameters
        ----------
        metric: str
            Metric name. Either 'mape' or 'smape'.
        """
        assert metric in ['mape', 'smape'], 'Please provide mape or smape'

        if metric == 'mape':
            return self.mape_forecasts
        else:
            return self.smape_forecasts

def get_base_data(train: Union[Tourism],
                  test: Union[Tourism],
                  info: Union[TourismInfo],
                  add_forecasts: Optional[pd.DataFrame] = None) -> 'BaseData':
    """

    Parameters
    ----------
    train:
    test:
    info:
    add_forecasts: pd.DataFrame
        Additional forecasts to include.
    """
    logger.info(info.name)

    features = []
    forecasts = []
    ground_truth = []
    groups = {}
    mape_forecasts = []
    smape_forecasts = []

    for group in info.groups:
        logger.info(group.name)

        seasonality = group.seasonality
        meta_models = _meta_models(seasonality, info.bases)

        train_group = train.get_group(group.name).y
        ground_truth_group = test.get_group(group.name).y

        logger.info('Calculating features')
        features_group = tsfeatures_r(train_group, freq=seasonality)
        features_group = features_group.fillna(0)
        features_group = features_group.sort_values('unique_id')
        ids_group = features_group['unique_id'].unique()
        features.append(features_group)

        logger.info('Calculating forecasts')
        forecasts_group = ground_truth_group.drop('y', 1)
        if meta_models:
            models = BaseModelsTrainer(meta_models).fit(None, train_group)
            forecasts_group = models.predict(forecasts_group)
        forecasts_group = forecasts_group.query('unique_id in @ids_group')
        forecasts_group = forecasts_group.sort_values(['unique_id', 'ds'])
        
        if add_forecasts is not None:
            forecasts_group = forecasts_group.merge(add_forecasts,
                                                    how='left',
                                                    on=['unique_id', 'ds'])
        forecasts.append(forecasts_group)

        logger.info('Adding ground truth')
        ground_truth_group = ground_truth_group.query('unique_id in @ids_group')
        ground_truth_group = ground_truth_group.sort_values(['unique_id', 'ds'])
        ground_truth.append(ground_truth_group)

        logger.info('Calculating MAPE')
        mape_forecasts_group = evaluate_models(ground_truth_group,
                                               forecasts_group,
                                               metric=mape)
        mape_forecasts_group = mape_forecasts_group.query('unique_id in @ids_group')
        mape_forecasts_group = mape_forecasts_group.sort_values('unique_id')
        mape_forecasts.append(mape_forecasts_group)

        logger.info('Calculating SMAPE')
        smape_forecasts_group = evaluate_models(ground_truth_group,
                                                forecasts_group,
                                                metric=smape)
        smape_forecasts_group = smape_forecasts_group.query('unique_id in @ids_group')
        smape_forecasts_group = smape_forecasts_group.sort_values('unique_id')
        smape_forecasts.append(smape_forecasts_group)

        groups[group.name] = ids_group
        sleep(5)

    features = pd.concat(features).reset_index(drop=True).fillna(0)
    forecasts = pd.concat(forecasts).reset_index(drop=True)
    ground_truth = pd.concat(ground_truth).reset_index(drop=True)
    mape_forecasts = pd.concat(mape_forecasts).reset_index(drop=True)
    smape_forecasts = pd.concat(smape_forecasts).reset_index(drop=True)

    return BaseData(features=features, forecasts=forecasts, \
                    ground_truth=ground_truth, \
                    mape_forecasts=mape_forecasts, \
                    smape_forecasts=smape_forecasts, \
                    groups=groups)

def _meta_models(seasonality: int, models: Iterable) -> Dict:
    """Returns dict of models."""
    meta_models = {
        'auto_arima_forec': ARIMA(seasonality, stepwise=False, approximation=False),
        'ets_forec': ETS(seasonality),
        'nnetar_forec': NNETAR(seasonality),
        'tbats_forec': TBATS(seasonality),
        'stlm_ar_forec': STLMFFORMA(seasonality),
        'rw_drift_forec': RandomWalk(seasonality, drift=True),
        'theta_forec': ThetaF(seasonality),
        'naive_forec': NaiveR(seasonality),
        'snaive_forec': SeasonalNaiveR(seasonality),
        'naive2_forec': Naive2(seasonality)
    }

    meta_models = {key: value for key, value in meta_models.items() if key in models}

    return meta_models
