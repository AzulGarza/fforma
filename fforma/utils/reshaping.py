#!/usr/bin/env python
# coding: utf-8
import itertools

import numpy as np
import pandas as pd
import multiprocessing as mp
from dask import delayed, compute
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

def long_to_wide(long_df, cols_to_parse=None,
                 cols_wide=None,
                 threads=None):

    if threads is None:
        threads = mp.cpu_count()

    long_df_dask = dd.from_pandas(long_df.set_index('unique_id'), npartitions=threads)
    long_df_dask = long_df_dask.map_partitions(lambda x: x.sort_values('ds'))

    if cols_to_parse is None:
        cols_to_parse = set(long_df.columns) - {'unique_id'}

    if cols_wide is None:
        cols_wide = cols_to_parse

    assert len(cols_to_parse) == len(cols_wide), 'Cols to parse and cols wide must have the same len'

    df_list = []
    for new_col, col in zip(cols_wide, cols_to_parse):
        df = long_df_dask[col].groupby('unique_id').apply(lambda df: df.values)
        df = df.rename(new_col)
        df = df.to_frame()
        df_list.append(df)

    with ProgressBar():
        df_list = compute(*df_list)

    wide_df = pd.concat(df_list, 1).reset_index()

    return wide_df

def train_to_horizontal(X_df, y_df, x_cols=None, threads=8):
    if x_cols is None:
        x_cols = list(set(X_df.columns)-set(['unique_id','ds']))

    x_horizontal = long_to_wide(X_df, cols_to_parse=['ds', x_cols],
                                cols_wide=['ds', 'X'],
                                threads=threads)
    y_horizontal = long_to_wide(y_df, threads=threads)

    train_df = x_horizontal.merge(y_horizontal, on='unique_id', how='outer')

    assert (train_df['ds_x'].apply(len) == train_df['ds_y'].apply(len)).sum() == len(train_df), 'ds_x and ds_y not corresponding'

    train_df['ds'] = train_df['ds_x']

    train_df = train_df[['unique_id','X','y','ds']]

    return train_df

def wide_to_long(df, lst_cols=None, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res
