#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from fforma.experiments.datasets.business import Business
from fforma.metrics.numpy import pinball_loss, quantile_calibration


def _get_metric(metric: str) -> Callable:
    if metric == 'pinball':
        return pinball_loss
    elif metric == 'calibration':
        return quantile_calibration
    else:
        raise Exception(f'Unknown metric: {metric}')

def main(directory: str, group: str) -> None:
    logger.info('Reading dataset')
    ts = Business.load(directory, group)
    logger.info('Dataset readed')

    main_path = Path(directory) / 'business'
    forecasts_path =  main_path / 'base'
    evaluation_path = main_path / 'evaluation'
    if not evaluation_path.exists():
        evaluation_path.mkdir(exist_ok=True, parents=True)

    file = forecasts_path / f'forecasts-{group.lower()}-quantile.csv'

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

    class_models = ['q_ar', 'q_ar_naive', 'q_ar_trend', 'q_ar_naive_trend']

    results = []

    for metric in ['quantile', 'calibration']:
        for tau in [0.3, 0.5, 0.7, 0.9]:
            results_metric = {'quantile': tau, 'metric': metric}
            models = [f'q_ar_{tau}', f'q_ar_{tau}_naive',
                      f'q_ar_{tau}_trend', f'q_ar_{tau}_naive_trend']

            for model, class_model in zip(models, class_models):
                y_hat = forecasts[model].values

                loss = _get_metric(metric)(y, y_hat, tau=tau)
                results_metric[class_model] = loss

            results.append(results_metric)

    results = pd.DataFrame(results).round(2)
    results.to_csv(evaluation_path / f'quantile-evaluation-{group.lower()}.csv', index=False)

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
