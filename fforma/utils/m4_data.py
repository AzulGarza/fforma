#!/usr/bin/env python
# coding: utf-8

import os
from six.moves import urllib
import subprocess

import numpy as np
import pandas as pd

from src.base_models import Naive2


seas_dict = {'Hourly': {'seasonality': 24, 'input_size': 24,
                       'output_size': 48, 'freq': 'H'},
             'Daily': {'seasonality': 7, 'input_size': 7,
                       'output_size': 14, 'freq': 'D'},
             'Weekly': {'seasonality': 52, 'input_size': 52,
                        'output_size': 13, 'freq': 'W'},
             'Monthly': {'seasonality': 12, 'input_size': 12,
                         'output_size':18, 'freq': 'M'},
             'Quarterly': {'seasonality': 4, 'input_size': 4,
                           'output_size': 8, 'freq': 'Q'},
             'Yearly': {'seasonality': 1, 'input_size': 4,
                        'output_size': 6, 'freq': 'D'}}

SOURCE_URL = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/'


def maybe_download(filename, directory):
    """Download the data from M4's website, unless it's already here.

    Parameters
    ----------
    filename: str
        Filename of M4 data with format /Type/Frequency.csv. Example: /Test/Daily-train.csv
    directory: str
        Custom directory where data will be downloaded.
    """
    data_directory = directory + "/m4"
    train_directory = data_directory + "/Train/"
    test_directory = data_directory + "/Test/"

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    if not os.path.exists(train_directory):
        os.mkdir(train_directory)
    if not os.path.exists(test_directory):
        os.mkdir(test_directory)

    filepath = os.path.join(data_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        size = os.path.getsize(filepath)
        print('Successfully downloaded', filename, size, 'bytes.')

    return filepath

def m4_parser(dataset_name, directory, num_obs=1000000):
    """Transform M4 data into a panel.

    Parameters
    ----------
    dataset_name: str
        Frequency of the data. Example: 'Yearly'.
    directory: str
        Custom directory where data will be saved.
    num_obs: int
        Number of time series to return.
    """
    data_directory = directory + "/m4"
    train_directory = data_directory + "/Train/"
    test_directory = data_directory + "/Test/"
    freq = seas_dict[dataset_name]['freq']

    m4_info = pd.read_csv(data_directory+'/M4-info.csv', usecols=['M4id','category'])
    m4_info = m4_info[m4_info['M4id'].str.startswith(dataset_name[0])].reset_index(drop=True)

    # Train data
    train_path='{}{}-train.csv'.format(train_directory, dataset_name)

    train_df = pd.read_csv(train_path, nrows=num_obs)
    train_df = train_df.rename(columns={'V1':'unique_id'})

    train_df = pd.wide_to_long(train_df, stubnames=["V"], i="unique_id", j="ds").reset_index()
    train_df = train_df.rename(columns={'V':'y'})
    train_df = train_df.dropna()
    train_df['split'] = 'train'
    train_df['ds'] = train_df['ds']-1
    # Get len of series per unique_id
    len_series = train_df.groupby('unique_id').agg({'ds': 'max'}).reset_index()
    len_series.columns = ['unique_id', 'len_serie']

    # Test data
    test_path='{}{}-test.csv'.format(test_directory, dataset_name)

    test_df = pd.read_csv(test_path, nrows=num_obs)
    test_df = test_df.rename(columns={'V1':'unique_id'})

    test_df = pd.wide_to_long(test_df, stubnames=["V"], i="unique_id", j="ds").reset_index()
    test_df = test_df.rename(columns={'V':'y'})
    test_df = test_df.dropna()
    test_df['split'] = 'test'
    test_df = test_df.merge(len_series, on='unique_id')
    test_df['ds'] = test_df['ds'] + test_df['len_serie'] - 1
    test_df = test_df[['unique_id','ds','y','split']]

    df = pd.concat((train_df,test_df))
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    # Create column with dates with freq of dataset
    len_series = df.groupby('unique_id').agg({'ds': 'max'}).reset_index()
    dates = []
    for i in range(len(len_series)):
        len_serie = len_series.iloc[i,1]
        ranges = pd.date_range(start='1970/01/01', periods=len_serie, freq=freq)
        dates += list(ranges)
    df.loc[:,'ds'] = dates

    df = df.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
    df.drop(columns=['M4id'], inplace=True)
    df = df.rename(columns={'category': 'x'})

    X_train_df = df[df['split']=='train'].filter(items=['unique_id', 'ds', 'x'])
    y_train_df = df[df['split']=='train'].filter(items=['unique_id', 'ds', 'y'])
    X_test_df = df[df['split']=='test'].filter(items=['unique_id', 'ds', 'x'])
    y_test_df = df[df['split']=='test'].filter(items=['unique_id', 'ds', 'y'])

    X_train_df = X_train_df.reset_index(drop=True)
    y_train_df = y_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)
    y_test_df = y_test_df.reset_index(drop=True)

    return X_train_df, y_train_df, X_test_df, y_test_df

def naive2_predictions(dataset_name, directory, num_obs, y_train_df = None, y_test_df = None):
    """Computes Naive2 predictions.

    Parameters
    ----------
    dataset_name: str
        Frequency of the data. Example: 'Yearly'.
    directory: str
        Custom directory where data will be saved.
    num_obs: int
        Number of time series to return.
    y_train_df: DataFrame
        Y train set returned by m4_parser
    y_test_df: DataFrame
        Y test set returned by m4_parser
    """
    # Read train and test data
    if (y_train_df is None) or (y_test_df is None):
        _, y_train_df, _, y_test_df = m4_parser(dataset_name, directory, num_obs)

    seasonality = seas_dict[dataset_name]['seasonality']
    input_size = seas_dict[dataset_name]['input_size']
    output_size = seas_dict[dataset_name]['output_size']
    freq = seas_dict[dataset_name]['freq']

    print('Preparing {} dataset'.format(dataset_name))
    print('Preparing Naive2 {} dataset predictions'.format(dataset_name))

    # Naive2
    y_naive2_df = pd.DataFrame(columns=['unique_id', 'ds', 'y_hat'])

    # Sort X by unique_id for faster loop
    y_train_df = y_train_df.sort_values(by=['unique_id', 'ds'])
    # List of uniques ids
    unique_ids = y_train_df['unique_id'].unique()
    X_test = pd.DataFrame(range(output_size))
    # Panel of fitted models
    for unique_id in unique_ids:
        # Fast filter X and y by id.
        top_row = np.asscalar(y_train_df['unique_id'].searchsorted(unique_id, 'left'))
        bottom_row = np.asscalar(y_train_df['unique_id'].searchsorted(unique_id, 'right'))
        y_id = y_train_df[top_row:bottom_row]

        y_naive2 = pd.DataFrame(columns=['unique_id', 'ds', 'y_hat'])
        y_naive2['ds'] = pd.date_range(start=y_id.ds.max(),
                                   periods=output_size+1, freq=freq)[1:]
        y_naive2['unique_id'] = unique_id
        y_naive2['y_hat'] = Naive2(seasonality).fit(None, y_id.y.to_numpy()).predict(X_test)
        y_naive2_df = y_naive2_df.append(y_naive2)

    y_naive2_df = y_test_df.merge(y_naive2_df, on=['unique_id', 'ds'], how='left')
    y_naive2_df.rename(columns={'y_hat': 'y_hat_naive2'}, inplace=True)

    results_dir = directory + '/results'
    naive2_file = results_dir + '/{}-naive2predictions_{}.csv'.format(dataset_name, num_obs)
    y_naive2_df.to_csv(naive2_file, encoding='utf-8', index=None)

    return y_naive2_df

def prepare_m4_data(dataset_name, directory, num_obs):
    """Pipeline that obtains M4 times series, tranforms it and gets naive2 predictions.

    Parameters
    ----------
    dataset_name: str
        Frequency of the data. Example: 'Yearly'.
    directory: str
        Custom directory where data will be saved.
    num_obs: int
        Number of time series to return.
    """
    m4info_filename = maybe_download('M4-info.csv', directory)

    dailytrain_filename = maybe_download('Train/Daily-train.csv', directory)
    hourlytrain_filename = maybe_download('Train/Hourly-train.csv', directory)
    monthlytrain_filename = maybe_download('Train/Monthly-train.csv', directory)
    quarterlytrain_filename = maybe_download('Train/Quarterly-train.csv', directory)
    weeklytrain_filename = maybe_download('Train/Weekly-train.csv', directory)
    yearlytrain_filename = maybe_download('Train/Yearly-train.csv', directory)

    dailytest_filename = maybe_download('Test/Daily-test.csv', directory)
    hourlytest_filename = maybe_download('Test/Hourly-test.csv', directory)
    monthlytest_filename = maybe_download('Test/Monthly-test.csv', directory)
    quarterlytest_filename = maybe_download('Test/Quarterly-test.csv', directory)
    weeklytest_filename = maybe_download('Test/Weekly-test.csv', directory)
    yearlytest_filename = maybe_download('Test/Yearly-test.csv', directory)

    X_train_df, y_train_df, X_test_df, y_test_df = m4_parser(dataset_name, directory, num_obs)

    results_dir = directory + '/results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    naive2_file = results_dir + '/{}-naive2predictions_{}.csv'
    naive2_file = naive2_file.format(dataset_name, num_obs)

    if not os.path.exists(naive2_file):
        y_naive2_df = naive2_predictions(dataset_name, directory, num_obs, y_train_df, y_test_df)
    else:
        y_naive2_df = pd.read_csv(naive2_file)
        y_naive2_df['ds'] = pd.to_datetime(y_naive2_df['ds'])

    return X_train_df, y_train_df, X_test_df, y_naive2_df

def prepare_full_m4_data(directory):
    """
    """
    data = [prepare_m4_data(dataset, directory, 100_000) for dataset in seas_dict.keys()]

    data = zip(*data)
    data = list(data)

    X_train_df, y_train_df, X_test_df, y_test_df = [pd.concat(tup).reset_index(drop=True) for tup in data]

    return X_train_df, y_train_df, X_test_df, y_test_df
