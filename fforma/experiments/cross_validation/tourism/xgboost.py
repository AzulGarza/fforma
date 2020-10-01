#!/usr/bin/env python
# coding: utf-8

import argparse
from functools import partial
import logging
from pathlib import Path

from optuna import Trial
from multiprocessing import cpu_count
import pandas as pd

from fforma.experiments.cross_validation.common import CrossValidation
from fforma.meta_learner import MetaLearnerXGBoost
from fforma.metrics.numpy import mape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = ['nbeats_generic_mape_forec',
          'nbeats_interpretable_mape_forec',
          'nbeats_*_mape_forec']

DEFAULT_PARAMS = {'threads': cpu_count() - 1,
                  'verbose': False,
                  'random_seed': 1,
                  'benchmark': 'nbeats_*_mape_forec'}

SEED = 10


def params(trial: Trial):

    params = {'n_estimators': trial.suggest_int('n_estimators', 1, 250),
              'eta': trial.suggest_uniform('eta', 1e-3, 1.0),
              'max_depth': trial.suggest_int('max_depth', 6, 14),
              'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
              'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0)}

    return params

def hpo_xgboost(directory: str, n_splits: int, n_trials: int):

    path = Path(directory) / 'tourism'

    data_cv = pd.read_pickle(path / 'base' / 'base_cv.p')
    data_test = pd.read_pickle(path / 'base' / 'base_training.p')

    data_cv.mape_forecasts = data_cv.mape_forecasts[['unique_id'] + MODELS]
    data_cv.forecasts = data_cv.forecasts[['unique_id', 'ds'] + MODELS]
    data_test.forecasts = data_test.forecasts[['unique_id', 'ds'] + MODELS]

    logger.info('Starting HPO')
    cv_model = CrossValidation(meta_learner=MetaLearnerXGBoost,
                               params=params,
                               default_params=DEFAULT_PARAMS,
                               metric=mape,
                               n_splits=n_splits,
                               n_trials=n_trials,
                               random_seed=SEED)
    cv_model = cv_model.fit(data_cv)

    logger.info('Generating forecasts')
    forecasts = cv_model.predict(data_test)

    forecasts_dir = path / 'forecasts' / 'xgboost'
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    forecasts_file = forecasts_dir / 'forecast.p'
    forecasts.to_pickle(forecasts_file)

    logger.info(f'Forecast saved at {forecasts_file}')
