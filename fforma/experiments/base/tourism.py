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


def main(args):

    directory = args.directory

    train = Tourism.load(directory)
    test = Tourism.load(directory, training=False)
    train_cv, validation = train.split_validation()

    dir_meta_data = Path(directory) / 'tourism' / 'base'
    dir_meta_data.mkdir(parents=True, exist_ok=True)

    logger.info('Calculating base data for CV')
    base_data_cv = get_base_data(train_cv, validation, TourismInfo)
    pd.to_pickle(base_data_cv, dir_meta_data / 'base_cv.p')

    logger.info('Calculating base data for training')
    base_data = get_base_data(train, test, TourismInfo)
    pd.to_pickle(base_data_cv, dir_meta_data / 'base_training.p')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get base data for Tourism dataset')
    parser.add_argument("--directory", required=True, type=str,
                        help="directory where Tourism data will be downloaded")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main(args)
