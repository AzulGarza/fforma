#!/usr/bin/env python
# coding: utf-8

from dataclasses import  dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common import download_file, URL_NBEATS

SOURCE_URL = 'https://robjhyndman.com/data/27-3-Athanasopoulos1.zip'


@dataclass
class Yearly:
    seasonality: int = 1
    horizon: int = 4
    freq: str = 'D'
    rows: int = 2
    name: str = 'Yearly'

@dataclass
class Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = 'Q'
    rows: int = 3
    name: str = 'Quarterly'

@dataclass
class Monthly:
    seasonality: int = 12
    horizon: int = 24
    freq: str = 'M'
    rows: int = 3
    name: str = 'Monthly'

@dataclass
class TourismInfo:
    groups: Tuple = (Yearly, Quarterly, Monthly)
    bases: Tuple = ('nbeats_generic_mape_forec',
                    'nbeats_interpretable_mape_forec',
                    'nbeats_*_mape_forec')
    benchmark: str = 'nbeats_generic_mape_forec'
    name: str = 'Tourism'

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
        path = Path(directory) / 'tourism' / 'datasets'

        data = []
        groups = {}

        for group in TourismInfo.groups:
            if training:
                file = path / f'{group.name.lower()}_in.csv'
            else:
                file = path / f'{group.name.lower()}_oos.csv'

            df = pd.read_csv(file)
            groups[group.name] = df.columns.values

            dfs = []
            for col in df.columns:
                df_col = df[col]
                lenght = df_col[0].astype(int)
                skip_rows = group.rows

                df_col = df_col[skip_rows:lenght + skip_rows]
                df_col = df_col.rename('y').to_frame()
                df_col['unique_id'] = col

                dfs.append(df_col)

            df = pd.concat(dfs)
            df['ds'] = df.groupby('unique_id').cumcount() + 1

            data.append(df)

        data = pd.concat(data).reset_index(drop=True)[['unique_id', 'ds', 'y']]

        return Tourism(y=data, groups=groups, train_data=training)

    @staticmethod
    def download(directory: str) -> None:
        """Download Tourism Dataset."""
        path = Path(directory) / 'tourism' / 'datasets'
        download_file(path, SOURCE_URL, decompress=True)

    @staticmethod
    def download_nbeats_forecasts(directory: str) -> None:
        """Download nbeats forecasts for Tourism."""
        path = Path(directory) / 'tourism' / 'base' / 'nbeats'

        for kind in ['cv', 'training']:
            url_nbeats_tourism = URL_NBEATS + f'tourism_forecasts_{kind}.p'
            download_file(path, url_nbeats_tourism)

    def get_group(self, group: str) -> 'Tourism':
        """Filters group data.

        Parameters
        ----------
        group: str
            Group name.
        """
        assert group in self.groups, \
            f'Please provide a valid group: {", ".join(self.groups.keys())}'

        ids = self.groups[group]

        y = self.y.query('unique_id in @ids')

        return Tourism(y=y, groups={group: ids}, train_data=self.train_data)

    def split_validation(self) -> Tuple['Tourism']:
        """Splits training data in train/validation."""

        assert self.train_data, 'Use training data for split validation'

        train = []
        val = []

        for group in TourismInfo.groups:
            df_group = self.get_group(group.name).y
            train_group = df_group.groupby('unique_id').apply(lambda df: df.head(-group.horizon)).reset_index(drop=True)
            val_group = df_group.groupby('unique_id').tail(group.horizon)
            val_group['ds'] = val_group.groupby('unique_id').cumcount() + 1

            train.append(train_group)
            val.append(val_group)

        train = pd.concat(train).reset_index(drop=True)
        val = pd.concat(val).reset_index(drop=True)

        return Tourism(y=train, groups=self.groups, train_data=self.train_data), \
               Tourism(y=val, groups=self.groups, train_data=False)
