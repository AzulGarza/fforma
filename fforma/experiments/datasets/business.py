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
    ts = ts.copy()

    fix_dates = ['2018-11-19', '2018-12-25',
                 '2019-01-01', '2019-02-04', '2019-02-21',
                 '2019-03-18', '2019-04-19', '2019-05-01',
                 '2019-09-15', '2019-09-16', '2019-09-29',
                 '2019-10-06', '2019-11-17', '2019-12-01',
                 '2019-12-15', '2019-12-22', '2019-12-25',
                 '2019-12-28', '2019-12-29', '2020-01-01',
                 '2020-01-05', '2020-01-12', '2020-01-26',
                 '2020-02-02', '2020-02-03', '2020-02-09',
                 '2020-02-22', '2020-02-23']
    seasonality = 7

    seasonal_fix_dates = pd.to_datetime(fix_dates) - pd.Timedelta(days=seasonality)
    seasonal_fix_dates = [date.strftime('%Y-%m-%d') for date in seasonal_fix_dates]

    to_fix = [date for date in seasonal_fix_dates if date in fix_dates]
    while to_fix:
        seasonal_fix_dates = [(pd.to_datetime(date) - pd.Timedelta(days=seasonality)).strftime('%Y-%m-%d') \
                              if date in to_fix else date \
                              for date in seasonal_fix_dates]
        to_fix = [date for date in seasonal_fix_dates if date in fix_dates]


    def get_updated_ts(fix_date, date):
        """
        fix_date: this date fixes date.
        date: date to fix.
        """
        fixed_date = ts.query('ds == @fix_date') \
                       .replace({fix_date: date})

        to_fix = ts.query('ds == @date')
        equal_order = np.array_equal(fixed_date['unique_id'].values,
                                     to_fix['unique_id'].values)

        assert equal_order

        fixed_date.index = to_fix.index

        return fixed_date
        
    fixed_dates = [get_updated_ts(fix_date, date) \
                   for fix_date, date in zip(seasonal_fix_dates, fix_dates)]
    fixed_dates = pd.concat(fixed_dates)
    ts.update(fixed_dates)

    ts['ds'] = pd.to_datetime(ts['ds'])
    ts = ts.query('ds >= "2018-05-02"').reset_index(drop=True)

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
