#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import pandas as pd

from ESRNN.m4_data import prepare_m4_data, seas_dict
from ESRNN.utils_evaluation import evaluate_prediction_owa
from fforma.metrics import WeightedPinballLoss
from fforma.meta_learner import MetaLearnerNN
from fforma.meta_results_r_data import prepare_fforma_data
from fforma.utils import (
    FactorQuantileRegressionAveraging,
    LassoQuantileRegressionAveraging,
    evaluate_forecasts,
    freqs
)

def main(args):
    dataset_name = args.dataset
    directory = args.directory
    hyndman_data = directory + '/hyndman_data'

    if not os.path.exists(hyndman_data):
        os.mkdir(hyndman_data)
    # ## Meta results from R (hyndman, et al)
    feats_train, X_models_train, \
        y_models_train, feats_test, \
        X_models_test = prepare_fforma_data(hyndman_data, dataset_name=dataset_name) #dataset_name=None for all data

    h = seas_dict[dataset_name]['output_size']

    df = X_models_train.groupby(['unique_id']).size()
    uids = df[df == h].index

    X_models_train = X_models_train[X_models_train['unique_id'].isin(uids)]
    y_models_train = y_models_train[y_models_train['unique_id'].isin(uids)]
    feats_train = feats_train[feats_train['unique_id'].isin(uids)]

    feats_train = feats_train.set_index(['unique_id'])
    X_models_train = X_models_train.set_index(['unique_id', 'ds'])
    y_models_train = y_models_train.set_index(['unique_id', 'ds'])
    feats_test = feats_test.set_index('unique_id')
    X_models_test = X_models_test.set_index(['unique_id', 'ds'])

    _, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name=dataset_name,
                                                          directory=directory,
                                                          num_obs=100_000)

    ################################
    ############## LASSO QRA
    ###############################
    qra = LassoQuantileRegressionAveraging(.5)
    qra.fit(X_models_train, y_models_train)
    qra_preds = qra.predict(X_models_test).rename(columns={'p50': 'lqra'})


    ################################
    ############## FACTOR QRA
    ###############################
    fqra = FactorQuantileRegressionAveraging((.5,), 0.9)
    fqra.fit(X_models_train, y_models_train)
    fqra_preds = fqra.predict(X_models_test).rename(columns={'p50': 'fqra'})

    ##############################
    ############# QFFORMA
    ##############################
    #Setting model
    nn_params = {'layers': [200, 100, 50, 25, 10],
                 'dropout': 0.1,
                 'epochs': 180,
                 'batch_size': 2,
                 'learning_rate': 0.001,
                 'gradient_eps': 1e-8,
                 'weight_decay': 0,
                 'freq_of_test': 20,
                 'lr_scheduler_step_size': 200,
                 'lr_decay': 0.1,
                 'loss_function': WeightedPinballLoss(0.4)}

    model = MetaLearnerNN(nn_params, X_models_train, y_models_train, h,
                          y_train_df=y_train_df, predictions_test=X_models_test,
                          y_test_df=y_test_df,
                          naive_seasonality=freqs[dataset_name])

    model.fit(feats_train, features_test=feats_test, verbose_eval=False)

    print(f'Min owa {model.min_owa} reachead at epoch {model.min_epoch}')

    fforma_predictions = model.predict(feats_test)
    fforma_predictions = pd.DataFrame(fforma_predictions,
                                      index=feats_test.index,
                                      columns=X_models_test.columns)

    fforma_predictions = (fforma_predictions * X_models_test).sum(1)
    fforma_predictions = fforma_predictions.rename('qfforma').to_frame()


    #####################################
    ############# EVALUATION
    ###################################
    forecasts = pd.concat([qra_preds, fqra_preds, fforma_predictions], 1).reset_index()
    evaluation = evaluate_forecasts(dataset_name, forecasts, directory, 100_000)

    print(evaluation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test QFFORMA on M4 data')
    parser.add_argument("--dataset", required=True, type=str,
                      choices=['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Hourly', 'Daily'],
                      help="set of M4 time series to be tested")
    parser.add_argument("--directory", required=True, type=str,
                      help="directory where M4 data will be downloaded")

    args = parser.parse_args()

    main(args)
