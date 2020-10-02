#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .common import maybe_download_decompress

SOURCE_URL = 'https://forecasters.org/data/m3comp/M3C.xls'

NEEDED_DATA = ('raw/M3C.xls',)


@dataclass
class Yearly:
    seasonality: int = 1
    horizon: int = 6
    freq: str = 'D'
    sheet_name: str = 'M3Year'
    name: str = 'Yearly'

@dataclass
class Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = 'Q'
    sheet_name: str = 'M3Quart'
    name: str = 'Quarterly'

@dataclass
class Monthly:
    seasonality: int = 12
    horizon: int = 18
    freq: str = 'M'
    sheet_name: str = 'M3Month'
    name: str = 'Monthly'

@dataclass
class Other:
    seasonality: int = 1
    horizon: int = 8
    freq: str = 'D'
    sheet_name: str = 'M3Other'
    name: str = 'Other'

@dataclass
class M3Info:
    groups: Tuple = (Yearly, Quarterly, Monthly, Other)
    name: str = 'M3'

@dataclass
class M3:
    y: pd.DataFrame
    groups: Dict
    train_data: bool

    @staticmethod
    def load(directory: str, training: bool = True) -> 'M3':
        """
        Downloads and loads M3 data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        training: bool
            Wheter return training or testing data. Default True.
        """

        path = Path(directory) / 'm3'
        M3.download(path)

        path = path / 'raw'

        data = []
        groups = {}

        for group in M3Info.groups:
            df = pd.read_excel(path / 'M3C.xls', sheet_name=group.sheet_name)
            df = df.rename(columns={'Series': 'unique_id'})
            df['unique_id'] = [group.name[0] + str(i + 1) for i in range(len(df))]

            groups[group.name] = df['unique_id'].to_list()

            id_vars = list(df.columns[:6])

            df = pd.melt(df, id_vars=id_vars, var_name='ds', value_name='y')
            df = df.dropna().sort_values(['unique_id', 'ds']).reset_index(drop=True)

            df = df.filter(items=['unique_id', 'ds', 'y'])
            df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

            if training:
                df = df.groupby('unique_id').apply(lambda df: df.head(-group.horizon)).reset_index(drop=True)
            else:
                df = df.groupby('unique_id').tail(group.horizon)
                df['ds'] = df.groupby('unique_id').cumcount() + 1

            data.append(df)

        data = pd.concat(data).reset_index(drop=True)

        return M3(y=data, groups=groups, train_data=training)

    @staticmethod
    def download(directory: Path) -> None:
        """Download Tourism Dataset."""
        maybe_download_decompress(directory, SOURCE_URL, NEEDED_DATA)

    def get_group(self, group: str) -> 'M3':
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

        return M3(y=y, groups={group: ids}, train_data=self.train_data)

    def split_validation(self) -> Tuple['M3']:
        """Splits training data in train/validation."""

        assert self.train_data, 'Use training data for split validation'

        train = []
        val = []

        for group in M3Info.groups:
            df_group = self.get_group(group.name).y
            train_group = df_group.groupby('unique_id').apply(lambda df: df.head(-group.horizon)).reset_index(drop=True)
            val_group = df_group.groupby('unique_id').tail(group.horizon)
            val_group['ds'] = val_group.groupby('unique_id').cumcount() + 1

            train.append(train_group)
            val.append(val_group)

        train = pd.concat(train).reset_index(drop=True)
        val = pd.concat(val).reset_index(drop=True)

        return M3(y=train, groups=self.groups, train_data=self.train_data), \
               M3(y=val, groups=self.groups, train_data=False)
