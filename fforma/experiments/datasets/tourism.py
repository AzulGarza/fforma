#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .common import maybe_download_decompress

SOURCE_URL = 'https://robjhyndman.com/data/27-3-Athanasopoulos1.zip'

RAW_DIR = 'raw/decompressed_data'
NEEDED_DATA = (f'{RAW_DIR}/monthly_in.csv',
               f'{RAW_DIR}/monthly_oos.csv',
               f'{RAW_DIR}/quarterly_in.csv',
               f'{RAW_DIR}/quarterly_oos.csv',
               f'{RAW_DIR}/yearly_in.csv',
               f'{RAW_DIR}/yearly_oos.csv')


@dataclass
class Monthly:
    seasonality: int = 12
    horizon: int = 24
    freq: str = 'M'
    rows: int = 3
    name: str = 'Monthly'

@dataclass
class Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = 'Q'
    rows: int = 3
    name: str = 'Quarterly'

@dataclass
class Yearly:
    seasonality: int = 1
    horizon: int = 2
    freq: str = 'D'
    rows: int = 2
    name: str = 'Yearly'

@dataclass
class TourismInfo:
    groups: List = (Monthly, Quarterly, Yearly)

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

        for group in TourismInfo.groups:
            if training:
                file = path / f'{group.name.lower()}_in.csv'
            else:
                file = path / f'{group.name.lower()}_oos.csv'

            df = pd.read_csv(file)
            groups[group.name] = df.columns.values

            df = df[group.rows:].melt(var_name='unique_id', value_name='y').dropna()
            df['ds'] = df.groupby('unique_id').cumcount()

            data.append(df)

        data = pd.concat(data).reset_index(drop=True)[['unique_id', 'ds', 'y']]

        return Tourism(y=data, groups=groups, train_data=training)

    @staticmethod
    def download(directory: str) -> None:
        """Download Tourism Dataset."""
        maybe_download_decompress(Path(directory) / 'Tourism', SOURCE_URL, NEEDED_DATA)

    def get_group(self, group: str) -> pd.DataFrame:
        """
        Filters group data.

        Parameters
        ----------
        group: str
            Name of group.
        """
        assert group in self.groups, \
            f'Please provide a valid group: {", ".join(self.groups.keys())}'

        return self.y[self.y['unique_id'].isin(self.groups[group])]

    def split_validation(self) -> Tuple['Tourism']:
        """Splits training data in train/validation."""

        assert self.train_data, 'Use training data for split validation'

        train = []
        val = []

        for group in TourismInfo.groups:
            df_group = self.get_group(group.name)
            train_group = df_group.groupby('unique_id').apply(lambda df: df.head(-group.horizon)).reset_index(drop=True)
            val_group = df_group.groupby('unique_id').tail(group.horizon)
            train.append(train_group)
            val.append(val_group)

        train = pd.concat(train).reset_index(drop=True)
        val = pd.concat(val).reset_index(drop=True)

        return Tourism(y=train, groups=self.groups, train_data=self.train_data), \
               Tourism(y=val, groups=self.groups, train_data=False)
