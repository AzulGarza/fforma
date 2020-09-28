#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
from dataclasses import dataclass
from gc import collect
import logging
from typing import Dict, Union

import pandas as pd
from tsfeatures.tsfeatures_r import tsfeatures_r

from fforma.base.trainer import BaseModelsTrainer
from fforma.base import (Naive2, ARIMA, ETS, NNETAR, STLM, TBATS, STLMFFORMA,
                         RandomWalk, ThetaF, NaiveR, SeasonalNaiveR)
from fforma.experiments.datasets.tourism import TourismInfo, Tourism

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaseData:
    features: pd.DataFrame
    forecasts: pd.DataFrame
    ground_truth: pd.DataFrame
    errors: pd.DataFrame
    groups: Dict

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

        features = self.features.query('unique_id in @ids')
        forecasts = self.forecasts.query('unique_id in @ids')
        ground_truth = self.ground_truth.query('unique_id in @ids')

        return BaseData(features=features, forecasts=forecasts, \
                        ground_truth=ground_truth, errors=pd.DataFrame,
                        groups={group: ids})


def get_base_data(train: Union[Tourism],
                  test: Union[Tourism],
                  info: Union[TourismInfo]) -> 'BaseData':

    logger.info(info.name)

    features = []
    forecasts = []
    ground_truth = []
    groups = {}

    for group in info.groups:
        logger.info(group.name)

        seasonality = group.seasonality

        train_group = train.get_group(group.name).y
        ground_truth_group = test.get_group(group.name).y

        meta_models = {
            # 'auto_arima_forec': ARIMA(seasonality, stepwise=False, approximation=False),
            # 'ets_forec': ETS(seasonality),
            # 'nnetar_forec': NNETAR(seasonality),
            # 'tbats_forec': TBATS(seasonality),
            # 'stlm_ar_forec': STLMFFORMA(seasonality),
            # 'rw_drift_forec': RandomWalk(seasonality, drift=True),
            # 'theta_forec': ThetaF(seasonality),
            # 'naive_forec': NaiveR(seasonality),train
            # 'snaive_forec': SeasonalNaiveR(seasonality),
            'y_hat_naive2': Naive2(seasonality)
        }

        logger.info('Calculating features')
        features_group = tsfeatures_r(train_group, freq=seasonality, parallel=True)
        features_group = features_group.reset_index(drop=True)
        features_group = features_group.fillna(0)
        features_group = features_group.sort_values('unique_id')
        ids_group = features_group['unique_id'].unique()
        features.append(features_group)

        logger.info('Calculating forecasts')
        models = BaseModelsTrainer(meta_models).fit(None, train_group)
        forecasts_group = models.predict(ground_truth_group.drop('y', 1))
        forecasts_group = forecasts_group.query('unique_id in @ids_group')
        forecasts_group = forecasts_group.sort_values(['unique_id', 'ds'])
        forecasts.append(forecasts_group)

        ground_truth_group = ground_truth_group.query('unique_id in @ids_group')
        ground_truth_group = ground_truth_group.sort_values(['unique_id', 'ds'])
        ground_truth.append(ground_truth_group)

        #TODO calculate errors

        groups[group.name] = ids_group
        sleep(5)

    features = pd.concat(features).reset_index(drop=True)
    forecasts = pd.concat(forecasts).reset_index(drop=True)
    ground_truth = test.y.filter(items=['unique_id', 'ds', 'y'])

    return BaseData(features=features, forecasts=forecasts, \
                    ground_truth=ground_truth, errors=pd.DataFrame(), \
                    groups=groups)
