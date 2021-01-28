#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from functools import partial
from gc import collect
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from time import time
from tsfeatures import tsfeatures

from fforma.base.trainer import BaseModelsTrainer
from fforma.base import Naive2, ARIMA, ETS, NNETAR, STLM, TBATS, STLMFFORMA, \
                        RandomWalk, ThetaF, NaiveR, SeasonalNaiveR
from fforma.experiments.datasets.business import Business, BusinessInfo


def _transform_base_file(file: str,
                         models: Iterable[str],
                         feats_to_drop: List[str]):
    """Transforms base file."""

    meta = pd.read_pickle(file)

    # Forecasts handling
    forecasts = meta.pop('forecasts') \
                    .filter(items=['unique_id', 'ds'] + list(models)) \
                    .assign(train_cutoff=meta['train_cutoff']) \
                    .replace([np.inf, -np.inf], np.nan)
    forecasts['naive2_forec'] = forecasts['naive2_forec'].fillna(forecasts['naive_forec'])
    for model in models:
        forecasts[model] = forecasts[model].clip(lower=0)

    if forecasts.isna().values.mean() > 0:
        raise Exception(f'NAN forecasts found on {file}, please check.')

    # Feautures handling
    features = meta.pop('features')  \
                   .drop(feats_to_drop, 1) \
                   .assign(train_cutoff = meta['train_cutoff']) \
                   .fillna(0) # Hyndman assumption
                              # https://github.com/robjhyndman/M4metalearning/blob/61ddc7101680e9df7219c359587d0b509d2b50d6/R/generate_classif_problem.R#L67

    if features.isna().values.mean() > 0:
        raise Exception(f'NAN features found on {file}, please check.')

    return meta, forecasts, features

def main(directory: str, group: str, replace: bool) -> None:
    logger.info('Reading dataset')
    ts = Business.load(directory, group)
    logger.info('Dataset readed')
    seasonality = BusinessInfo[group].seasonality

    main_path = Path(directory) / 'business'
    saving_path = main_path / f'fforma_{group}'
    saving_path.mkdir(exist_ok=True, parents=True)
    base_path = main_path / 'base'
    base_path.mkdir(exist_ok=True, parents=True)

    # Meta models
    meta_models = {'auto_arima_forec': ARIMA(seasonality),
                   'ets_forec': ETS(seasonality),
                   'nnetar_forec': NNETAR(seasonality),
                   'tbats_forec': TBATS(seasonality),
                   'stlm_ar_forec': STLMFFORMA(seasonality),
                   'rw_drift_forec': RandomWalk(seasonality, drift=True),
                   'theta_forec': ThetaF(seasonality),
                   'naive_forec': NaiveR(seasonality),
                   'snaive_forec': SeasonalNaiveR(seasonality),
                   'naive2_forec': Naive2(seasonality),}

    periods = 91
    cutoffs = pd.date_range(end=ts['ds'].max(), periods=periods, freq='W-THU')

    for cutoff in cutoffs:
        logger.info(f'============Cutoff: {cutoff}')

        file = saving_path / f'cutoff={cutoff.date()}_freq={seasonality}.p'
        if file.exists() and not replace:
            logger.info('File already saved\n')
            continue

        test_cutoff = cutoff + pd.Timedelta(days=seasonality)
        train = ts.query('ds < @cutoff')
        test = ts.query('ds >= @cutoff & ds < @test_cutoff').drop('y', 1)

        logger.info('Features...')
        init = time()
        features = tsfeatures(train, seasonality)
        feats_time = time() - init
        logger.info(f'Features time: {feats_time}')

        logger.info('Training...')
        init = time()
        model = BaseModelsTrainer(meta_models)
        model.fit(None, train)
        training_time = time() - init
        logger.info(f'Training time: {training_time}')

        logger.info('Forecasting...')
        init = time()
        forecasts = model.predict(test)
        forecasting_time = time() - init
        logger.info(f'Forecasting time: {forecasting_time}\n')

        meta = {'features_time': feats_time,
                'training_time': training_time,
                'forecasting_time': forecasting_time,
                'train_cutoff': cutoff,
                'test_cutoff': test_cutoff,
                'features': features,
                'forecasts': forecasts,}

        pd.to_pickle(meta, file)

        del features, model, forecasts
        collect()

    logger.info(f'Forecast finished')

    feats_to_drop = ['series_length', 'nperiods',
                     'seasonal_period', 'hurst', 'entropy']

    transform = partial(_transform_base_file,
                        models=meta_models.keys(),
                        feats_to_drop=feats_to_drop)

    files = [saving_path / f'cutoff={cutoff.date()}_freq={seasonality}.p' \
             for cutoff in cutoffs]
    meta, forecasts, features = zip(*[transform(file) for file in files])

    meta = pd.DataFrame(meta).sort_values('test_cutoff')
    forecasts = pd.concat(forecasts).filter(items=forecasts_cols)
    features = pd.concat(features)

    meta.to_csv(base_path / f'meta-{group.lower()}.csv', index=False)
    forecasts.to_csv(base_path / f'forecasts-{group.lower()}.csv', index=False)
    features.to_csv(base_path / f'features-{group.lower()}.csv', index=False)

    logger.info('Results saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Base Forecasts for Business Datasets')
    parser.add_argument('--directory', required=True, type=str,
                        help='experiments directory')
    parser.add_argument('--group', required=True, type=str,
                        help='group (GLB or BRC)',
                        choices=['GLB', 'BRC'])
    parser.add_argument('--replace', required=False, action='store_true',
                        help='Replace files already saved')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.group, args.replace)
