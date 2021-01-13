#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from functools import reduce
from pathlib import Path
from time import time
from typing import Callable, Tuple

import pandas as pd

from ..datasets.business import Business
from fforma.meta_learner import MetaLearnerMean, MetaLearnerSoftMin, \
                                MetaLearnerBestModel, MetaLearnerXGBoost
from fforma.utils.evaluation import evaluate_models
from fforma.metrics.numpy import mae, mape, rmse, smape, pinball_loss


def _get_metric(metric: str) -> Callable:
    if metric == 'rmse':
        return rmse
    elif metric == 'mape':
        return mape
    elif metric == 'smape':
        return smape
    elif metric == 'mae':
        return mae
    elif metric == 'pinball':
        return pinball_loss
    else:
        raise Exception(f'Unknown metrc: {metric}')

def main(directory: str, group: str, metric: str) -> None:
    logger.info('Reading dataset')
    ts = Business.load(directory, group)
    logger.info('Dataset readed')

    metric_fun = _get_metric(metric)

    main_path = Path(directory) / 'business'
    saving_path = main_path / f'fforma_{group}'
    base_path = main_path / 'base'
    forecasts_path = main_path / 'forecasts'
    forecasts_path.mkdir(exist_ok=True, parents=True)

    # TODO common.py with dataclass to download this datasets.
    meta = pd.read_csv(base_path / f'meta-{group.lower()}.csv')
    forecasts = pd.read_csv(base_path / f'forecasts-{group.lower()}.csv')
    forecasts['ds'] = pd.to_datetime(forecasts['ds'])
    features = pd.read_csv(base_path / f'features-{group.lower()}.csv')
    meta['prev_train_cutoff'] = meta['train_cutoff'].shift(1)

    # optimal params by hyndman
    optimal_params = {'n_estimators': 94,
                      'eta': 0.58,
                      'max_depth': 14,
                      'subsample': 0.92,
                      'colsample_bytree': 0.77}
    n_estimators = optimal_params.pop('n_estimators')
    benchmark = 'naive2_forec'
    random_seed = 1


    for i, (prev_cutoff, cutoff) in enumerate(zip(meta['prev_train_cutoff'], meta['train_cutoff'])):
        logger.info(f'============Pct: {100 * i / meta.shape[0]}')

        file = saving_path / f'forecasts_cutoff={cutoff}_metric={metric}.p'
        if file.exists():
            logger.info('File already saved\n')
            continue

        if i == 0:
            continue
        #print(prev_id_)
        logger.info('Wrangling')
        init = time()
        forecasts_train = forecasts.query('train_cutoff in @prev_cutoff') \
                                   .drop('train_cutoff', 1)
        forecasts_train = forecasts_train.merge(ts, how='left', on=['unique_id', 'ds'])
        constant = 2 if metric in ['mape', 'smape'] else 0
        errors_train = evaluate_models(forecasts_train.filter(items=['unique_id', 'ds', 'y']),
                                       forecasts_train.drop('y', 1),
                                       metric_fun)

        errors_train = errors_train.fillna(100)
        features_train = features.query('train_cutoff in @prev_cutoff') \
                                 .drop('train_cutoff', 1)
        # Test datasets
        forecasts_test = forecasts.query('train_cutoff in @cutoff') \
                                  .drop('train_cutoff', 1)
        features_test = features.query('train_cutoff in @cutoff') \
                                .drop('train_cutoff', 1)
        wrangling_time = time() - init
        logger.info(f'Wrangling time: {wrangling_time}')

        ############# Train MetaLearners
        ######### Classic FFORMA
        logger.info('Fforma')
        init = time()
        meta_learner = MetaLearnerXGBoost(optimal_params, benchmark, n_estimators, random_seed)
        meta_learner = meta_learner.fit(features_train, errors_train)
        fforma_forecasts = meta_learner.predict(features_test, forecasts_test)
        fforma_time = time() - init
        logger.info(f'Fforma time: {fforma_time}')

        ######### Mean
        logger.info('Mean')
        init = time()
        meta_learner = MetaLearnerMean()
        meta_learner = meta_learner.fit(forecasts_test)
        mean_forecasts = meta_learner.predict()
        mean_time = time() - init
        logger.info(f'Mean time: {mean_time}')

        ######### Softmin
        logger.info('Softmin')
        init = time()
        meta_learner = MetaLearnerSoftMin()
        meta_learner = meta_learner.fit(errors_train, forecasts_test)
        softmin_forecasts = meta_learner.predict()
        softmin_time = time() - init
        logger.info(f'Softmin time: {softmin_time}')

        ######### BestModel
        logger.info('Best model')
        init = time()
        meta_learner = MetaLearnerBestModel()
        meta_learner = meta_learner.fit(errors_train, forecasts_test)
        best_model_forecasts = meta_learner.predict()
        best_model_time = time() - init
        logger.info(f'Best model time: {best_model_time}')

        info = {'wrangling_time': wrangling_time,
                'fforma_time': fforma_time,
                'mean_time': mean_time,
                'softmin_time': softmin_time,
                'best_model_time': best_model_time,
                'fforma_forecasts': fforma_forecasts,
                'mean_forecasts': mean_forecasts,
                'softmin_forecasts': softmin_forecasts,
                'best_model_forecasts': best_model_forecasts}

        pd.to_pickle(info, file)

    logger.info(f'Forecast finished')

    regexp = f'forecasts_*_metric={metric}.p'
    ensemble_forecasts = []
    for model in ['fforma', 'mean', 'softmin', 'best_model']:
        forecasts = [pd.read_pickle(file)[f'{model}_forecasts'] for file in saving_path.glob(regexp)]
        forecasts = pd.concat(forecasts)
        forecasts.rename({'y_hat': f'{model}_ensemble'}, axis=1, inplace=True)
        ensemble_forecasts.append(forecasts)

    ensemble_forecasts = reduce(lambda x, y: pd.merge(x, y, on=['unique_id', 'ds']),
                                ensemble_forecasts)

    file = forecasts_path / f'ensemble_forecasts_group={group}_metric={metric}.csv'
    ensemble_forecasts.to_csv(file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPO')
    parser.add_argument('--directory', required=True, type=str,
                        help='experiments directory')
    parser.add_argument('--group', required=True, type=str,
                        help='group (GLB or BRC)',
                        choices=['GLB', 'BRC'])
    parser.add_argument('--metric', required=True, type=str,
                        help='Metric respect to optimize',
                        choices=['mae', 'mape', 'smape', 'pinball', 'rmse'])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.group, args.metric)
