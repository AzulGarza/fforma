#!/usr/bin/env python
# coding: utf-8

import argparse
from functools import partial
import logging
from pathlib import Path

from optuna import Trial
import pandas as pd

from fforma.experiments.cross_validation.common import CrossValidation
from fforma.meta_learner import MetaLearnerFFNN
from fforma.metrics.numpy import mape
from fforma.metrics.pytorch import mape_loss, pinball_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = ['nbeats_generic_mape_forec',
          'nbeats_interpretable_mape_forec',
          'nbeats_*_mape_forec']

DEFAULT_PARAMS = {'device': 'cpu',
                  'display_step': 1,
                  'verbose': False,
                  'random_seed': 1,
                  'benchmark': 'nbeats_*_mape_forec'}

SEED = 10


def params(trial: Trial):
    params = {'n_epochs': trial.suggest_categorical('n_epochs', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
              'lr': trial.suggest_loguniform('lr', 1e-2, 1e-1),
              'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 1024]),
              'dropout': trial.suggest_loguniform('dropout', 1e-2, 1.0),
              'layers': trial.suggest_categorical('layers', [[400, 200, 100, 50, 25], [512], [512, 256, 128], [256, 256], [256, 128, 64]]),
              'softmax': trial.suggest_categorical('softmax', [True, False]),
              'loss': trial.suggest_categorical('loss', ['pinball', 'mape'])}

    if params['loss'] == 'pinball':
        params['quantile'] = trial.suggest_uniform('quantile', 0.45, 0.65)
        params['loss_function'] = partial(pinball_loss, tau=params['quantile'])
    elif params['loss'] == 'mape':
        params['loss_function'] = mape_loss

    return params

def hpo_fnn(directory: str, n_splits: int, n_trials: int):

    path = Path(directory) / 'tourism'

    data_cv = pd.read_pickle(path / 'base' / 'base_cv.p')
    data_test = pd.read_pickle(path / 'base' / 'base_training.p')

    data_cv.forecasts = data_cv.forecasts[['unique_id', 'ds'] + MODELS]
    data_test.forecasts = data_test.forecasts[['unique_id', 'ds'] + MODELS]

    logger.info('Starting HPO')
    cv_model = CrossValidation(meta_learner=MetaLearnerFFNN,
                               params=params,
                               default_params=DEFAULT_PARAMS,
                               metric=mape,
                               n_splits=n_splits,
                               n_trials=n_trials,
                               random_seed=SEED)
    cv_model = cv_model.fit(data_cv)

    logger.info('Generating forecasts')
    forecasts = cv_model.predict(data_test)

    forecasts_dir = path / 'forecasts' / 'fnn'
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    forecasts_file = forecasts_dir / 'forecast.p'
    forecasts.to_pickle(forecasts_file)

    logger.info(f'Forecast saved at {forecasts_file}')
