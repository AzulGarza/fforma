#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fforma.experiments.datasets.business import Business
from fforma.experiments.business.ensemble_forecasts import _get_metric

def main(directory: str, group: str) -> None:
    logger.info('Reading dataset')
    ts = Business.load(directory, group)
    logger.info('Dataset readed')

    forecasts_path = Path(directory) / 'business' / 'forecasts'
    metrics = ['mae', 'mape', 'smape', 'rmse', 'pinball']

    results = []
    for metric in metrics:
        constant = 2 if metric in ['mape', 'smape'] else 0
        results_metric = {'metric': metric}
        file = forecasts_path / f'ensemble_forecasts_group={group}_metric={metric}.csv'

        if not file.exists():
            y = None
            y_hat = None
        else:
            forecasts = pd.read_csv(file)
            forecasts['ds'] = pd.to_datetime(forecasts['ds'])
            min_ds = forecasts['ds'].min()
            forecasts = ts.merge(forecasts, how='left', on=['unique_id', 'ds']) \
                          .query('ds >= @min_ds')
            y = forecasts['y'].values

        for model in ['fforma', 'mean', 'softmin', 'best_model']:
            y_hat = forecasts[f'{model}_ensemble'].values
            loss = _get_metric(metric)(y, y_hat) if y is not None else np.nan
            results_metric[f'{model}_ensemble'] = loss

        results.append(results_metric)
    results = pd.DataFrame(results).round(2)

    print(results.to_latex(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for Business Datasets')
    parser.add_argument('--directory', required=True, type=str,
                        help='experiments directory')
    parser.add_argument('--group', required=True, type=str,
                        help='group (GLB or BRC)',
                        choices=['GLB', 'BRC'])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.group)
