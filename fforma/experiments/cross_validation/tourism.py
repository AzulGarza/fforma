#!/usr/bin/env python
# coding: utf-8

from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple

from multiprocessing import cpu_count
from optuna import Trial
import pandas as pd

from fforma.experiments.datasets.tourism import TourismInfo
from fforma.meta_learner import MetaLearnerFFNN, MetaLearnerXGBoost
from fforma.metrics.numpy import mape
from fforma.metrics.pytorch import mape_loss, pinball_loss

DEFAULT_PARAMS_FFNN = {'device': 'cpu',
                       'display_step': 1,
                       'verbose': False,
                       'random_seed': 1,
                       'benchmark': TourismInfo.benchmark}

DEFAULT_PARAMS_XGBOOST = {'threads': cpu_count() - 1,
                          'verbose': False,
                          'random_seed': 1,
                          'benchmark': TourismInfo.benchmark}


def tourism_params(directory: str, model: str) -> Tuple:
    path = Path(directory) / 'tourism'

    data_cv = pd.read_pickle(path / 'base' / 'base_cv.p')
    data_test = pd.read_pickle(path / 'base' / 'base_training.p')

    if model == 'ffnn':
        return data_cv, data_test, MetaLearnerFFNN, params_ffnn, \
               DEFAULT_PARAMS_FFNN, mape, \
               DEFAULT_PARAMS_FFNN['random_seed'], path
    elif model == 'xgboost':
        return data_cv, data_test, MetaLearnerXGBoost, params_xgboost, \
               DEFAULT_PARAMS_XGBOOST, mape, \
               DEFAULT_PARAMS_XGBOOST['random_seed'], path
    else:
        raise Exception(f'Unknown model: {model}')

def params_ffnn(trial: Trial):
    """FFNN parameters."""
    params = {'n_epochs': trial.suggest_int('n_epochs', 1, 100),
              'lr': trial.suggest_loguniform('lr', 1e-2, 1e-1),
              'batch_size': trial.suggest_categorical('batch_size',
                                                      [32, 64, 128, 1024]),
              'dropout': trial.suggest_loguniform('dropout', 1e-2, 1.0),
              'layers': trial.suggest_categorical('layers',
                                                  [[400, 200, 100, 50, 25],
                                                   [512],
                                                   [512, 256, 128],
                                                   [256, 256],
                                                   [256, 128, 64]]),
              'softmax': trial.suggest_categorical('softmax', [True, False]),
              'loss': trial.suggest_categorical('loss', ['pinball', 'mape'])}

    if params['loss'] == 'pinball':
        params['quantile'] = trial.suggest_uniform('quantile', 0.45, 0.65)
        params['loss_function'] = partial(pinball_loss, tau=params['quantile'])
    elif params['loss'] == 'mape':
        params['loss_function'] = mape_loss

    return params

def params_xgboost(trial: Trial):
    """XGBoost parameters."""
    params = {'n_estimators': trial.suggest_int('n_estimators', 1, 250),
              'eta': trial.suggest_uniform('eta', 1e-3, 1.0),
              'max_depth': trial.suggest_int('max_depth', 6, 14),
              'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
              'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0)}

    return params
