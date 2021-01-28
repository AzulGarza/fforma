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
from fforma.base import QuantileAutoRegression
from fforma.experiments.datasets.business import Business, BusinessInfo


def _transform_base_file(file: str,
                         models: Iterable[str]):
    """Transforms base file."""

    meta = pd.read_pickle(file)

    # Forecasts handling
    forecasts = meta.pop('forecasts') \
                    .assign(train_cutoff=meta['train_cutoff']) \
                    .replace([np.inf, -np.inf], np.nan)

    if forecasts.isna().values.mean() > 0:
        raise Exception(f'NAN forecasts found on {file}, please check.')

    return meta, forecasts

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
    ar_terms = [7, 14, 28]
    for tau in [0.3, 0.5, 0.7, 0.9]:
        meta_models[f'q_ar_{tau}'] = QuantileAutoRegression(tau=tau,
                                                            ar_terms=ar_terms,
                                                            max_diffs=0)
        meta_models[f'q_ar_{tau}_naive'] = QuantileAutoRegression(tau=tau,
                                                                  ar_terms=ar_terms,
                                                                  naive_forecasts=True,
                                                                  max_diffs=0)
        meta_models[f'q_ar_{tau}_trend'] = QuantileAutoRegression(tau=tau,
                                                                  ar_terms=ar_terms,
                                                                  add_trend=True)
        meta_models[f'q_ar_{tau}_naive_trend'] = QuantileAutoRegression(tau=tau,
                                                                        ar_terms=ar_terms,
                                                                        naive_forecasts=True,
                                                                        add_trend=True)

    periods = 54
    cutoffs = pd.date_range(end=ts['ds'].max(), periods=periods, freq='W-THU')

    for cutoff in cutoffs:
        logger.info(f'============Cutoff: {cutoff}')

        file = saving_path / f'cutoff={cutoff.date()}_freq={seasonality}_quantile.p'
        if file.exists() and not replace:
            logger.info('File already saved\n')
            continue

        test_cutoff = cutoff + pd.Timedelta(days=seasonality)
        train = ts.query('ds < @cutoff')
        test = ts.query('ds >= @cutoff & ds < @test_cutoff').drop('y', 1)

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

        meta = {'training_time': training_time,
                'forecasting_time': forecasting_time,
                'train_cutoff': cutoff,
                'test_cutoff': test_cutoff,
                'features': features,
                'forecasts': forecasts,}

        pd.to_pickle(meta, file)

        del features, model, forecasts
        collect()

    logger.info(f'Forecast finished')

    transform = partial(_transform_base_file,
                        models=meta_models.keys())

    files = [saving_path / f'cutoff={cutoff.date()}_freq={seasonality}_quantile.p' \
             for cutoff in cutoffs]
    meta, forecasts = zip(*[transform(file) for file in files])
    meta = pd.DataFrame(meta).sort_values('test_cutoff')
    forecasts = pd.concat(forecasts)

    meta.to_csv(base_path / f'meta-{group.lower()}-quantile.csv', index=False)
    forecasts.to_csv(base_path / f'forecasts-{group.lower()}-quantile.csv', index=False)

    logger.info('Results saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantile Forecasts for Business Datasets')
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
