#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

from .common import maybe_download_decompress

seas_dict = {'Monthly': {'seasonality': 12,
                         'horizon': 24, 'freq': 'M',
                         'rows': 3},
             'Quarterly': {'seasonality': 4,
                           'horizon': 8, 'freq': 'Q',
                           'rows': 3},
             'Yearly': {'seasonality': 1,
                        'horizon': 4, 'freq': 'D',
                        'rows': 2}}

SOURCE_URL = 'https://robjhyndman.com/data/27-3-Athanasopoulos1.zip'

MAIN_DIR = '/tourism/raw/decompressed_data'
NEEDED_DATA = (f'{MAIN_DIR}/monthly_in.csv',
               f'{MAIN_DIR}/monthly_oos.csv',
               f'{MAIN_DIR}/quarterly_in.csv',
               f'{MAIN_DIR}/quarterly_oos.csv',
               f'{MAIN_DIR}/yearly_in.csv',
               f'{MAIN_DIR}/yearly_oos.csv')


@dataclass
class Tourism:
    y: pd.DataFrame
    groups: Dict
    train_data: bool

    @staticmethod
    def load(directory: str, training: bool = True) -> 'Tourism':
        """
        Downloads and loads Tourism data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        training: bool
            Wheter return training or testing data. Default True.
        """

        Tourism.download(directory)

        path = Path('data/tourism/raw/decompressed_data')

        data = []
        groups = {}

        for dataset_name in seas_dict.keys():
            seasonality =  seas_dict[dataset_name]['seasonality']
            rows = seas_dict[dataset_name]['rows']

            if training:
                file = path/f'{dataset_name.lower()}_in.csv'
            else:
                file = path/f'{dataset_name.lower()}_oos.csv'

            df = pd.read_csv(file)

            groups[dataset_name] = df.columns.values

            df = df[rows:].melt(var_name='unique_id', value_name='y').dropna()
            df['ds'] = df.groupby('unique_id').cumcount()

            data.append(df)

        data = pd.concat(data).reset_index(drop=True)[['unique_id', 'ds', 'y']]

        return Tourism(y=data, groups=groups, train_data=training)

    @staticmethod
    def download(directory: str) -> None:
        """Download Tourism."""
        maybe_download_decompress(directory, SOURCE_URL, NEEDED_DATA)

    def get_group(self, group: str) -> pd.DataFrame:
        """
        Filters group data.

        Parameters
        ----------
        group: str
            Name of group.
        """
        assert group in self.groups, \
            f'Please provide a valid group: {", ".join(self.groups.values)}'

        return self.y[self.y['unique_id'].isin(self.groups[group])]

    def split_validation(self) -> Tuple['Tourism']:
        """Splits training data in train/validation."""

        assert self.train_data, 'Use training data for split validation'

        train = []
        val = []
        for group in self.groups:
            horizon = seas_dict[group]['horizon']
            df_group = self.get_group(group)
            train_group = df_group.groupby('unique_id').apply(lambda df: df.head(-horizon)).reset_index(drop=True)
            val_group = df_group.groupby('unique_id').tail(horizon)
            train.append(train_group)
            val.append(val_group)

        train = pd.concat(train).reset_index(drop=True)
        val = pd.concat(val).reset_index(drop=True)

        return Tourism(y=train, groups=self.groups, train_data=self.train_data), \
               Tourism(y=val, groups=self.groups, train_data=False)
