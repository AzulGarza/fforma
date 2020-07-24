#!/usr/bin/env python
# coding: utf-8

import os
import requests

import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import deepcopy
from src.utils import long_to_wide, wide_to_long
from tsfeatures.metrics import evaluate_panel
from dask import delayed, compute
from dask.diagnostics import ProgressBar

META_URL = 'https://github.com/FedericoGarza/meta-data/releases/download/v.0.0.1/'

###############################################################################
########## UTILS FOR FFORMA FLOW
###############################################################################

def calc_errors(preds_df, y_panel_df, y_insample_df, seasonality, benchmark_model):
    """Calculates OWA of each time series
    usign benchmark_model as benchmark.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    y_insample_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        Train set.
    seasonality: int
        Frequency of the time seires.
    benchmark_model: str
        Column name of the benchmark model.

    Returns
    -------
    Pandas DataFrame
        OWA errors for each time series and each model.
    """

    assert benchmark_model in y_panel_df.columns

    y_hat_panel_fun = lambda model_name: preds_df[['unique_id','ds', model_name]].rename(columns={model_name: 'y_hat'})

    errors_smape = y_panel_df[['unique_id']].drop_duplicates().reset_index(drop=True)
    errors_mase = errors_smape.copy()

    model_names = set(preds_df.columns) - {'unique_id', 'ds'}

    for model_name in model_names:
        errors_smape[model_name] = None
        errors_mase[model_name] = None
        y_hat_panel = y_hat_panel_fun(model_name)

        errors_smape[model_name] = evaluate_panel(y_test=y_panel_df, y_hat=y_hat_panel,
                                                  y_train=y_insample_df,
                                                  metric=smape,
                                                  seasonality=seasonality)['error']
        errors_mase[model_name] = evaluate_panel(y_test=y_panel_df, y_hat=y_hat_panel,
                                                 y_train=y_insample_df,
                                                 metric=mase,
                                                 seasonality=seasonality)['error']

    mean_smape_benchmark = errors_smape[benchmark_model].mean()
    mean_mase_benchmark = errors_mase[benchmark_model].mean()

    errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
    errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

    errors = errors_mase / mean_mase_benchmark + errors_smape / mean_smape_benchmark
    errors = 0.5 * errors

    return errors

def calc_errors_widing(preds_df, y_panel_df, y_insample_df, seasonality, benchmark_model):
    """Calculates OWA of each time series
    usign benchmark_model as benchmark.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    y_insample_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        Train set.
    seasonality: pandas df
        Frequency of the time seires.
    benchmark_model: str
        Column name of the benchmark model.

    Returns
    -------
    Pandas DataFrame
        OWA errors for each time series and each model.
    """
    df_wide = [long_to_wide(df).set_index(['unique_id']) \
               for df in (preds_df, y_panel_df.rename(columns={'y': 'y_test'}),
                          y_insample_df.rename(columns={'y': 'y_train'}))]

    df_wide = pd.concat(df_wide, axis=1).join(seasonality.set_index('unique_id'))

    assert benchmark_model in df_wide.columns

    df_wide_fun = lambda model_name: df_wide.rename(columns={model_name: 'y_hat'})

    errors_smape = y_panel_df[['unique_id']].drop_duplicates().reset_index(drop=True)
    errors_mase = errors_smape.copy()

    model_names = set(preds_df.columns) - {'unique_id', 'ds'}
    model_names.add(benchmark_model)

    for model_name in model_names:
        errors_smape[model_name] = None
        errors_mase[model_name] = None
        df_wide_hat = df_wide_fun(model_name)

        errors_smape[model_name] = evaluate_wide_panel(wide_panel=df_wide_hat,
                                                       metric=smape)['smape']
        errors_mase[model_name] = evaluate_wide_panel(wide_panel=df_wide_hat,
                                                      metric=mase)['mase']

    mean_smape_benchmark = errors_smape[benchmark_model].mean()
    mean_mase_benchmark = errors_mase[benchmark_model].mean()

    errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
    errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

    errors = errors_smape / mean_smape_benchmark + errors_mase / mean_mase_benchmark
    errors = 0.5 * errors

    return errors

def calc_errors_wide(y_panel_wide_df, y_hat_col='y_hat'):
    """
    y_panel_wide_df: pandas DF
        columns ['unique_id', 'seasonality', 'y_insample', 'y_test']

    returns smape and mase
    """

    uids = []
    metrics = []
    losses = []

    for uid, df in y_panel_wide_df.groupby('unique_id'):
        seasonality = df['seasonality'].item()
        y_insample = df['y_insample'].item()
        y_test = df['y_test'].item()
        y_hat = df[y_hat_col].item()

        loss_smape = delayed(smape)(y_test, y_hat)
        uids.append(uid)
        metrics.append(f'smape_{y_hat_col}')
        losses.append(loss_smape)

        loss_mase = delayed(mase)(y_test, y_hat, y_insample, seasonality)
        uids.append(uid)
        metrics.append(f'mase_{y_hat_col}')
        losses.append(loss_mase)

    with ProgressBar():
        losses = compute(*losses)

    dict_losses = {'unique_id': uids, 'metric': metrics, 'loss': losses}
    df_losses = pd.DataFrame.from_dict(dict_losses)
    df_losses = df_losses.set_index(['unique_id', 'metric']).unstack()
    df_losses.columns = df_losses.columns.droplevel(0).rename('')
    df_losses = df_losses.reset_index()

    return df_losses

def smape_mase_naive2_from_long(y_insample_df, y_test_df):
    y_insample_wide = long_to_wide(y_insample_df.drop('ds', 1)).rename(columns={'y': 'y_insample'})
    y_test_wide = long_to_wide(y_test_df.drop('ds', 1)).rename(columns={'y': 'y_test'})

    complete_wide = y_insample_wide.merge(y_test_wide, how='left', on=['unique_id'])
    complete_wide['seasonality'] = complete_wide['unique_id'].apply(lambda x: FREQ_DICT[x[0]])

    errors = calc_errors_wide(complete_wide, 'y_hat_naive2')
    complete_wide = complete_wide.merge(errors, how='left', on=['unique_id'])

    return complete_wide

def evaluate_wide_panel(wide_panel, metric, remove_na=True):

    metric_name = metric.__name__

    losses = []
    uids = []
    for uid, df in wide_panel.groupby('unique_id'):
        y_test = df['y_test'].item()
        y_test = np.array(y_test)

        y_hat = df['y_hat'].item()
        y_hat = np.array(y_hat)

        if remove_na:
            y_test = y_test[~np.isnan(y_test)]
            y_hat = y_hat[~np.isnan(y_hat)]

        if metric_name in ['mase', 'rmsse']:
            y_train = df['y_train'].item()
            y_train = np.array(y_train)

            if remove_na:
                y_train = y_train[~np.isnan(y_train)]

            seasonality = df['seasonality'].item()

            loss = delayed(metric)(y=y_test, y_hat=y_hat,
                                   y_train=y_train,
                                   seasonality=seasonality)
        else:
            loss = delayed(metric)(y_test, y_hat)

        losses.append(loss)
        uids.append(uid)

    with ProgressBar():
        losses = compute(*losses)

    loss_panel = [uids, losses]
    loss_panel = zip(*loss_panel)

    loss_panel = pd.DataFrame(loss_panel,
                              columns=['unique_id', metric_name])

    return loss_panel

def evaluate_fforma_experiment(long_preds, directory, kind='M4', threads=None):
    filepath = f'{directory}/{kind}_errors_naive2.p'

    if not os.path.exists(filepath):
        filename = filepath.split('/')[-1]
        URL = META_URL + filename
        r = requests.get(URL, stream=True)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(filepath, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)

        t.close()

        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")

        size = os.path.getsize(filepath)
        print('Successfully downloaded', filename, size, 'bytes.')

    errors_naive2 = pd.read_pickle(filepath)

    #predictions
    predictions = long_to_wide(long_preds, threads=threads)

    #Preparing to evaluation
    complete_data = predictions.merge(errors_naive2, how='left', on=['unique_id'])
    errors = calc_errors_wide(complete_data)
    errors = complete_data.merge(errors, how='left', on=['unique_id'])

    errors = errors.loc[:, errors.columns.str.contains('mase|smape')]
    errors = errors.mean().to_dict()
    errors = pd.DataFrame(errors, index=[0])

    for metric in ['mase', 'smape']:
        errors[f'{metric}_rel'] = errors[f'{metric}_y_hat'] / errors[f'{metric}_y_hat_naive2']

    errors['owa'] = 0.5 * (errors['mase_rel'] + errors['smape_rel'])

    model_owa, model_mase, model_smape = [errors[col].item() for col in ('owa', 'mase_y_hat', 'smape_y_hat')]

    print("OWA: {:03.3f}".format(model_owa))
    print("MASE: {:03.3f}".format(model_mase))
    print("SMAPE: {:03.3f}".format(100 * model_smape))

    return errors

def get_prediction_panel(y_panel_df, h, freq):
    """Constructs panel to use with
    predict method.
    """
    df = y_complete_train_df[['unique_id', 'ds']].groupby('unique_id').max().reset_index()

    predict_panel = []
    for idx, df in df.groupby('unique_id'):
        date = df['ds'].values.item()
        unique_id = df['unique_id'].values.item()

        date_range = pd.date_range(date, periods=h, freq=freq)
        df_ds = pd.DataFrame.from_dict({'ds': date_range})
        df_ds['unique_id'] = unique_id
        predict_panel.append(df_ds[['unique_id', 'ds']])

    predict_panel = pd.concat(predict_panel)

    return predict_panel

def smape(y, y_hat):
    """
    Calculates Symmetric Mean Absolute Percentage Error.
    y: numpy array
    actual test values
    y_hat: numpy array
    predicted values
    return: sMAPE
    """
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
    return smape

def mase(y, y_hat, y_train, seasonality):
    """
    Calculates Mean Absolute Scaled Error.
    y: numpy array
    actual test values
    y_hat: numpy array
    predicted values
    y_train: numpy array
    actual train values for Naive1 predictions
    seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
    return: MASE
    """
    y_hat_naive = []
    for i in range(seasonality, len(y_train)):
        y_hat_naive.append(y_train[(i - seasonality)])

    masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
    if masep==0:
        print('y_train', y_train)
        print('y_hat_naive', y_hat_naive)

    mase = np.mean(abs(y - y_hat)) / masep
    return mase
