#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from pathlib import Path

import pandas as pd

from fforma.meta_learner import MetaLearnerMean, MetaLearnerMedian


def main(directory: str, dataset: str) -> None:

    path = Path(directory) / dataset.lower()
    base_data = pd.read_pickle(path / 'base'/ 'base_training.p')

    benchmark_path = path / 'benchmarks'
    benchmark_path.mkdir(parents=True, exist_ok=True)

    logger.info('Computing mean benchmark')
    benchmark_mean = MetaLearnerMean().fit(base_data.forecasts).predict()
    benchmark_mean.rename({'y_hat': 'median_ensemble_forec'}, axis=1, inplace=True)

    logger.info('Computing median benchmark')
    benchmark_median = MetaLearnerMedian().fit(base_data.forecasts).predict()
    benchmark_median.rename({'y_hat': 'mean_ensemble_forec'}, axis=1, inplace=True)

    benchmarks = benchmark_mean.merge(benchmark_median,
                                      how='left',
                                      on=['unique_id', 'ds'])

    benchmarks.to_pickle(benchmark_path / 'benchmarks.p')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get benchmark forecasts')
    parser.add_argument('--directory', required=True, type=str,
                        help='Experiments directory')
    parser.add_argument('--dataset', required=True, type=str,
                        help='Either Tourism or M3')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.dataset)
