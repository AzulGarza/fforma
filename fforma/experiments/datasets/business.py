import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import s3fs
from dotenv import load_dotenv

from .common import Info

load_dotenv()


def cleanear_brc(ts: pd.DataFrame, seasonality: int) -> pd.DataFrame:
    """
    Cleans BRC dataset.
    """

    ts['ds'] = pd.to_datetime(ts['ds'])

    return ts

def cleanear_glb(ts: pd.DataFrame, seasonality: int) -> pd.DataFrame:
    """
    Cleans GLB dataset.
    """
    ts = ts.copy()
    fix_dates = ['2019-04-30', '2019-05-01']
    seasonality = 7

    seasonal_fix_dates = pd.to_datetime(fix_dates) - pd.Timedelta(days=seasonality)
    seasonal_fix_dates = [date.strftime('%Y-%m-%d') for date in seasonal_fix_dates]

    fixed_dates = ts.query('ds in @seasonal_fix_dates') \
                    .replace(dict(zip(seasonal_fix_dates, fix_dates)))
    fixed_dates.index = ts.query('ds in @fix_dates').index

    ts.update(fixed_dates)

    ts['ds'] = pd.to_datetime(ts['ds'])

    return ts

@dataclass
class BRC:
    seasonality: int = 7
    horizon: int = 7
    cleaner: Callable[[pd.DataFrame, int], pd.DataFrame] = cleanear_brc

@dataclass
class GLB:
    seasonality: int = 7
    horizon: int = 7
    cleaner: Callable[[pd.DataFrame, int], pd.DataFrame] = cleanear_glb

BusinessInfo = Info(groups=('BRC', 'GLB'),
                    class_groups=(BRC, GLB))

class Business:

    @staticmethod
    def load(directory: str,
             group: str,
             return_weekly: bool = False):
        """
        Downloads and loads Tourism data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'GLB', 'BRC'.

        Notes
        -----
        [1] Returns train+test sets.
        """
        path = Path(directory) / 'business' / 'datasets'

        Business.download(directory, group)

        class_group = BusinessInfo[group]

        df = pd.read_csv(path / f'ts-{group.lower()}.csv')
        df = class_group.cleaner(df, class_group.seasonality)

        if return_weekly:
            df = df.groupby(['unique_id', pd.Grouper(key='ds', freq='W-THU')]) \
                   .sum() \
                   .reset_index()

            min_ds, max_ds = df['ds'].agg(['min', 'max'])
            df = df.query('ds > @min_ds & ds < @max_ds')

        return df

    @staticmethod
    def download(directory: str, group: str) -> None:
        """Downloads Business Dataset."""

        fs = s3fs.S3FileSystem(key=os.environ['AWS_ACCES_KEY_ID'],
                               secret=os.environ['AWS_SECRET_ACCESS_KEY'])

        path = Path(directory) / 'business' / 'datasets'
        path.mkdir(parents=True, exist_ok=True)

        download_file = path / f'ts-{group.lower()}.csv'
        if not download_file.exists():
            file = f'research-storage-orax/business-data/ts-{group.lower()}.csv'
            fs.download(file, str(download_file))
