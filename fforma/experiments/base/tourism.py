#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fforma.experiments.datasets.tourism import TourismInfo, Tourism
from .common import get_base_data


def main(directory: str, training: bool):

    train = Tourism.load(directory)

    if training:
        test = Tourism.load(directory, training=False)
        label = 'training'
    else:
        train, test = train.split_validation()
        label = 'cv'

    dir_meta_data = Path(directory) / 'tourism' / 'base'
    dir_meta_data.mkdir(parents=True, exist_ok=True)

    logger.info(f'Calculating base data for {label}')
    base_data = get_base_data(train, test, TourismInfo)
    pd.to_pickle(base_data, dir_meta_data / f'base_{label}.p')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get base data for Tourism dataset')
    parser.add_argument("--directory", required=True, type=str,
                        help="directory where Tourism data will be downloaded")
    parser.add_argument('--training', default=False, action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.training)
