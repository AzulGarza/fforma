#!/usr/bin/env python
# coding: utf-8

import argparse
from functools import partial
import logging
from pathlib import Path
from typing import Tuple

from optuna import Trial
import pandas as pd

from .common import CrossValidation
from .tourism import tourism_params

def _dataset_params(directory: str, dataset: str, model: str) -> Tuple:
    if dataset == 'tourism':
        return tourism_params(directory, model)

def main(directory: str, dataset: str, model: str,
         n_splits: int, n_trials: int) -> None:
    data_cv, data_test, meta_learner, params, \
        default_params, metric, seed, path = _dataset_params(directory, dataset, model)

    logger.info('Starting HPO')
    cv_model = CrossValidation(meta_learner=meta_learner,
                               params=params,
                               default_params=default_params,
                               metric=metric,
                               n_splits=n_splits,
                               n_trials=n_trials,
                               random_seed=seed)
    cv_model = cv_model.fit(data_cv)

    logger.info('Generating forecasts')
    forecasts = cv_model.predict(data_test)

    forecasts_dir = path / 'forecasts' / model
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    forecasts_file = forecasts_dir / 'forecast.p'
    forecasts.to_pickle(forecasts_file)

    logger.info(f'Forecast saved at {forecasts_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPO')
    parser.add_argument('--directory', required=True, type=str,
                        help='experiments directory')
    parser.add_argument('--dataset', required=True, type=str,
                        help='dataset (tourism or m3)',
                        choices=['m3', 'tourism'])
    parser.add_argument('--model', required=True, type=str,
                        help='Model to cv-train (ffnn or xgboost)',
                        choices=['ffnn', 'xgboost'])
    parser.add_argument('--n_splits', required=True, type=int,
                        help='number of folds for kfold cv')
    parser.add_argument('--n_trials', required=True, type=int,
                        help='number of hpo trials')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.dataset, args.model, args.n_splits, args.n_trials)
