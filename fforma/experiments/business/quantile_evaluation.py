#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fforma.experiments.datasets.business import Business
from fforma.metrics.numpy import pinball_loss


def main(directory: str, group: str) -> None:
    logger.info('Reading dataset')
    ts = Business.load(directory, group)
    ts = pd.concat(ts)
    logger.info('Dataset readed')

    main_path = Path(directory) / 'business'
    forecasts_path =  main_path / 'base'
    evaluation_path = main_path / 'evaluation'
    if not evaluation_path.exists():
        evaluation_path.mkdir(exist_ok=True, parents=True)

    file = forecasts_path / f'forecasts-group={group.lower()}-quantile.csv'

    if not file.exists():
        raise Exception('File does not exist')

    forecasts = pd.read_csv(file)
    if forecasts.isna().values.mean() > 0:
        raise Exception('Some forecasts are NA, check procedure')
        
    forecasts['ds'] = pd.to_datetime(forecasts['ds'])
    min_ds = forecasts['ds'].min()
    forecasts = ts.merge(forecasts, how='left', on=['unique_id', 'ds']) \
                  .query('ds >= @min_ds')
    y = forecasts['y'].values

    results = []
    for tau in [0.3, 0.5, 0.7, 0.9]:
        results_metric = {'quantile': tau}

        for model in [f'q_ar_{tau}', f'q_ar_{tau}_naive',
                      f'q_ar_{tau}_trend', f'q_ar_{tau}_naive_trend']:
            y_hat = forecasts[model].values

            loss = pinball_loss(y, y_hat, tau=tau)
            results_metric[metric] = loss

        results.append(results_metric)

    results = pd.DataFrame(results).round(2)
    results.to_csv(evaluation_path / f'quantile_evaluation_{group}.csv', index=False)

    print(results.to_latex(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantile Evaluation for Business Datasets')
    parser.add_argument('--directory', required=True, type=str,
                        help='experiments directory')
    parser.add_argument('--group', required=True, type=str,
                        help='group (GLB or BRC)',
                        choices=['GLB', 'BRC'])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.group)
