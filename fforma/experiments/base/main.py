#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from .common import get_base_data, URL_NBEATS
from fforma.experiments.datasets.common import maybe_download_decompress
from fforma.experiments.datasets.tourism import Tourism, TourismInfo
from fforma.experiments.datasets.m3 import M3, M3Info


def _dataset(dataset: str):

    if dataset == 'tourism':
        return Tourism, TourismInfo
    elif dataset == 'm3':
        return M3, M3Info
    else:
        raise Exception(f'Unknown dataset: {dataset}')

def main(directory: str, dataset: str, training: bool) -> None:

    data_class, info_class = _dataset(dataset)

    train = data_class.load(directory)

    if training:
        test = data_class.load(directory, training=False)
        label = 'training'
    else:
        train, test = train.split_validation()
        label = 'cv'

    dir_meta_data = Path(directory) / dataset / 'base'
    dir_meta_data.mkdir(parents=True, exist_ok=True)

    logger.info('Downloading nbeats data')
    dir_meta_data_nbeats = dir_meta_data / 'nbeats'
    dir_meta_data_nbeats.mkdir(parents=True, exist_ok=True)
    file_nbeats = f'{dataset}_forecasts_{label}.p'
    url_nbeats = URL_NBEATS + file_nbeats
    needed_data = ('raw/' + file_nbeats,)
    maybe_download_decompress(dir_meta_data_nbeats, url_nbeats, needed_data)
    forecasts_nbeats = pd.read_pickle(dir_meta_data_nbeats / 'raw' / file_nbeats)

    logger.info(f'Calculating base data for {label}')
    base_data = get_base_data(train, test, info_class, forecasts_nbeats)
    pd.to_pickle(base_data, dir_meta_data / f'base_{label}.p')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get base data for Tourism or M3 dataset')
    parser.add_argument('--directory', required=True, type=str,
                        help='directory where data will be downloaded')
    parser.add_argument('--dataset', required=True, type=str,
                        help='dataset to get base data',
                        choices=['tourism', 'm3'])
    parser.add_argument('--training', default=False, action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.dataset, args.training)
