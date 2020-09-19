#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd

from tsfeatures.tsfeatures_r import tsfeatures_r_wide
from tscompdata import m3, tourism
from rpy2.robjects.packages import importr
from src.meta_model import MetaModels
from src.base_models import Naive2
from src.base_models_r import (
    ARIMA,
    ETS,
    NNETAR,
    TBATS,
    STLM,
    STLMFFORMA,
    RandomWalk,
    ThetaF,
    Naive,
    SeasonalNaive
)

def split_holdout(series):
    h = series['horizon']
    y = series['y']

    y_train, y_val = y[:-h], y[-h:]

    return y_train, y_val

def ds_holdout(series):
    h = series['horizon']
    ds = np.arange(1, h + 1)

    return ds

def main(args):
    directory = args.directory
    kind = args.kind

    if kind == 'M3':
        data = m3.get_complete_wide_data(directory)
    elif kind == 'TOURISM':
        data = tourism.get_complete_wide_data(directory)

    y_train_val = [split_holdout(row) for idx, row in data.iterrows()]
    data['y_train'], data['y_val'] = zip(*y_train_val)

    data['ds'] = [ds_holdout(row) for idx, row in data.iterrows()]

    stats = importr('stats')

    meta_models = {
         'auto_arima_forec': lambda freq: ARIMA(freq, stepwise=False, approximation=False),
         'ets_forec': ETS,
         'nnetar_forec': NNETAR,
         'tbats_forec': TBATS,
         'stlm_ar_forec': lambda freq: STLMFFORMA(freq),
         'rw_drift_forec': lambda freq: RandomWalk(freq=freq, drift=True),
         'theta_forec': ThetaF,
         'naive_forec': Naive,
         'snaive_forec': SeasonalNaive,
         'y_hat_naive2': Naive2,
    }

    print('Validation meta data')

    validation_data = data[['unique_id', 'ds', 'horizon', 'seasonality', 'y_train', 'y_val']].rename(columns={'y_train': 'y'})
    vaidation_models = MetaModels(meta_models).fit(validation_data)
    validation_preds = vaidation_models.predict(validation_data.drop(['seasonality', 'y'], 1))

    validation_features = tsfeatures_r_wide(validation_data, parallel=True).reset_index()

    print('Test meta data')

    test_data = data[['unique_id', 'ds', 'horizon', 'seasonality', 'y', 'y_test']]
    test_models = MetaModels(meta_models).fit(test_data)
    test_preds = test_models.predict(test_data.drop(['seasonality', 'y'], 1))

    test_features = tsfeatures_r_wide(test_data, parallel=True).reset_index()

    save_data = (validation_data, validation_features, validation_preds, test_data, test_features, test_preds)

    print('Saving data')

    dir_meta_data = f'{directory}/meta-data'
    if not os.path.exists(dir_meta_data):
        os.mkdir(dir_meta_data)

    pd.to_pickle(save_data, f'{dir_meta_data}/{kind}-meta-data.pickle')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get metadata for M3 or TOURISM')
    parser.add_argument("--directory", required=True, type=str,
                        help="directory where M3 data will be downloaded")

    parser.add_argument("--kind", required=True, type=str,
                        help="M3 or TOURISM")

    args = parser.parse_args()

    main(args)
