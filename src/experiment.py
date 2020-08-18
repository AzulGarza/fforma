import os
import glob
import yaml
import argparse
import itertools
import ast
import pickle
import time

import numpy as np
import pandas as pd

from src.benchmarks import (
    LassoQuantileRegressionAveraging,
    FactorQuantileRegressionAveraging
)

#############################################################################
# EXPERIMENT SPECS
#############################################################################

DICT_FREQS = {'H':24, 'D': 7, 'W':52, 'M': 12, 'Q': 4, 'Y': 1}

GRID_QRA1 = {'model_type': ['qra'],
             'tau': [0.5, 0.45, 0.55],
             'penalty': [1., 1.5, 2., 2.5, 3, 3.5],
             'grid_id': ['grid_qra1']}

GRID_QRA2 = {'model_type': ['qra'],
             'tau': [0.5, 0.48, 0.49, 0.51, 0.52],
             'penalty': [3, 3.5, 4, 4.5, 5, 5.5],
             'grid_id': ['grid_qra2']}

GRID_QRA3 = {'model_type': ['qra'],
             'tau': [0.5, 0.48, 0.49, 0.51, 0.52],
             'penalty': [5, 10, 15, 20, 25],
             'grid_id': ['grid_qra3']}

GRID_QRA4 = {'model_type': ['qra'],
             'tau': [0.5, 0.48, 0.49, 0.51, 0.52],
             'penalty': [25, 30, 35, 40, 45],
             'grid_id': ['grid_qra4']}

GRID_QRA5 = {'model_type': ['qra'],
             'tau': [0.5, 0.48, 0.49, 0.51, 0.52],
             'penalty': [31, 32,33, 34, 35, 36, 37, 38, 35],
             'grid_id': ['grid_qra5']}



GRID_FQRA1 = {'model_type': ['fqra'],
              'tau': [0.45, 0.48, 0.5, 0.53, 0.55],
              'n_components': [1],
              'grid_id': ['grid_fqra1']} # for timeseries without insufficient obs

GRID_FFORMA1 = {'model_type': ['fforma'],
                'n_estimators': np.arange(1, 250, 1),
                'eta': np.arange(0.01, 1, 0.01),
                'max_depth': np.arange(6, 15, 1),
                'subsample': np.arange(0.5, 1, 0.1),
                'colsample_bytree': np.arange(0.5, 1, 0.1),
                'grid_id': ['grid_fforma1']}

GRID_FFORMA2 = {'model_type': ['fforma'],
                'n_estimators': np.arange(1, 20, 1),
                'eta': np.arange(0.01, 1, 0.01),
                'max_depth': np.arange(6, 15, 1),
                'subsample': np.arange(0.5, 1, 0.1),
                'colsample_bytree': np.arange(0.5, 1, 0.1),
                'grid_id': ['grid_fforma2']}

GRID_FFORMA3 = {'model_type': ['fforma'],
                'n_estimators': [1, 2, 3],
                'eta': np.arange(0.57, 0.9, 0.01),
                'max_depth': [6, 7, 8, 9],
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'grid_id': ['grid_fforma3']}

GRID_FFORMA4 = {'model_type': ['fforma'],
                'n_estimators': [1, 2, 3],
                'eta': np.arange(0.8, 1.3, 0.01),
                'max_depth': [6, 7, 8, 9],
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'grid_id': ['grid_fforma4']}

GRID_FFORMA5 = {'model_type': ['fforma'],
                'n_estimators': [1, 2, 3],
                'eta': np.arange(1.1, 2, 0.01),
                'max_depth': [6, 7, 8, 9],
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'grid_id': ['grid_fforma5']}

GRID_FFORMA6 = {'model_type': ['fforma'],
                'n_estimators': [1, 2, 3],
                'eta': np.arange(1.9, 3, 0.01),
                'max_depth': [6, 7, 8, 9],
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'grid_id': ['grid_fforma6']}

GRID_FFORMA7 = {'model_type': ['fforma'],
                'n_estimators': np.arange(1, 100, 1),
                'eta': np.arange(0.01, 2, 0.05),
                'max_depth': np.arange(6, 15, 1),
                'subsample': np.arange(0.5, 1, 0.1),
                'colsample_bytree': np.arange(0.5, 1, 0.1),
                'grid_id': ['grid_fforma7']}

GRID_FFORMA8 = {'model_type': ['fforma'],
                'n_estimators': np.arange(1, 3, 1),
                'eta': np.arange(1.8, 4, 0.05),
                'max_depth': np.arange(6, 15, 1),
                'subsample': np.arange(0.1, 0.5, 0.1),
                'colsample_bytree': np.arange(0.5, 1, 0.1),
                'grid_id': ['grid_fforma8']}


GRID_FFORMAM4 = {'model_type': ['fforma'],
                'n_estimators': [94],
                'eta': [0.58],
                'max_depth': [14],
                'subsample': [0.92],
                'colsample_bytree': [0.77],
                'grid_id': ['grid_fforma_m4']}

GRID_QFFORMAM3 = {'model_type': ['qfforma'],
                  'n_epochs' : [5, 10, 15],
                  'lr': [1e-3, 5e-3, 7e-3],
                  'batch_size': [64, 128],
                  'gradient_eps': [1e-8],
                  'weight_decay': [0],
                 #'lr_scheduler_step_size': [10],
                  'lr_decay': [0.0, 0.5, 1],
                  'dropout': [0, 0.3, 0.5],
                  'layers': ['[200, 100, 50, 25, 10]'],
                  'use_softmax': [False, True],
                  'train_percentile': [0.45, 0.48, 0.5, 0.53, 0.55],
                  'display_step': [1],
                  'random_seed': [1],
                  'grid_id': ['grid_qfforma_m3']}

GRID_QFFORMA1 = {'model_type': ['qfforma'],
                 'n_epochs' : [5, 20, 50],
                 'lr': [5e-5, 7e-5, 1e-4, 5e-4, 7e-4, 1e-3, 5e-3, 7e-3, 1e-2, 5e-2],
                 'batch_size': [64, 128],
                 'gradient_eps': [1e-8],
                 'weight_decay': [0, 1, 2, 5],
                 #'lr_scheduler_step_size': [10],
                 'lr_decay': [0.5, 1],
                 'dropout': [0, 0.3],
                 'layers': ['[100]', '[100, 50]', '[200, 100, 50, 25, 10]'],
                 'use_softmax': [False],
                 'train_percentile': [0.4, 0.5, 0.6],
                 'display_step': [1],
                 'random_seed': [1],
                 'grid_id': ['grid_qfforma1']}

GRID_QFFORMA2 = {'model_type': ['qfforma'],
                 'n_epochs' : [51, 101, 201],
                 'lr': [5e-5, 7e-5, 1e-4, 5e-4, 7e-4],
                 'batch_size': [64],
                 'gradient_eps': [1e-8],
                 'weight_decay': [0],
                 #'lr_scheduler_step_size': [10],
                 'lr_decay': [0.5, 0.8, 1],
                 'dropout': [0, 0.3],
                 'layers': ['[200, 100, 50, 25, 10]', '[400, 200, 100, 50, 25]'],
                 'use_softmax': [True],
                 'train_percentile': [0.45, 0.48, 0.49, 0.5, 0.51, 0.55],
                 'display_step': [10],
                 'random_seed': [1],
                 'grid_id': ['grid_qfforma2']}

GRID_QFFORMA3 = {'model_type': ['qfforma'],
                 'n_epochs' : [5],
                 'lr': [5e-5, 7e-5],
                 'batch_size': [64],
                 'gradient_eps': [1e-8],
                 'weight_decay': [0],
                 #'lr_scheduler_step_size': [10],
                 'lr_decay': [0.5, 0.8, 1],
                 'dropout': [0, 0.3],
                 'layers': ['[400, 200, 100, 50, 25]'],
                 'use_softmax': [True, False],
                 'train_percentile': [0.45, 0.5, 0.51, 0.55],
                 'display_step': [1],
                 'random_seed': [1],
                 'grid_id': ['grid_qfforma3']}

GRID_QFFORMA4 = {'model_type': ['qfforma'],
                 'n_epochs' : [5, 10],
                 'lr': [1e-2, 1e-3, 7e-5],
                 'batch_size': [64],
                 'gradient_eps': [1e-8],
                 'weight_decay': [0],
                 #'lr_scheduler_step_size': [10],
                 'lr_decay': [0.5, 0.8, 1],
                 'dropout': [0.0, 0.1],
                 'layers': ['[512, 256, 128, 64, 32, 16, 8, 4, 2]', '[400, 200, 100, 50, 25]'],
                 'use_softmax': [True],
                 'train_percentile': [0.45, 0.5, 0.51, 0.55],
                 'display_step': [1],
                 'random_seed': [1],
                 'grid_id': ['grid_qfforma4']}

GRID_QFFORMA5 = {'model_type': ['qfforma'],
                 'n_epochs' : [10, 15, 20, 25],
                 'lr': [1e-2],
                 'batch_size': [64],
                 'gradient_eps': [1e-8],
                 'weight_decay': [0],
                 #'lr_scheduler_step_size': [10],
                 'lr_decay': [0.5, 0.8, 1],
                 'dropout': [0.0],
                 'layers': ['[400, 200, 100, 50, 25]'],
                 'use_softmax': [True],
                 'train_percentile': [0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58],
                 'display_step': [1],
                 'random_seed': [1],
                 'grid_id': ['grid_qfforma5']}

GRID_QFFORMA6 = {'model_type': ['qfforma'],
                 'n_epochs' : [5, 10],
                 'lr': [1e-2, 5e-2, 0.1, 1e-3],
                 'batch_size': [64],
                 'gradient_eps': [1e-8],
                 'weight_decay': [0, 0.1, 0.3, 0.5],
                 #'lr_scheduler_step_size': [10],
                 'lr_decay': [1],
                 'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                 'layers': ['[512, 256, 128, 64, 32, 16, 8, 4, 2]'],
                 'use_softmax': [True],
                 'train_percentile': [0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58],
                 'display_step': [1],
                 'random_seed': [1],
                 'grid_id': ['grid_qfforma6']}

GRID_QFFORMATEST = {'model_type': ['qfforma'],
                     'n_epochs' : [5, 10],
                     'lr': [1e-5, 5e-5, 7e-5],
                     'batch_size': [64],
                     'gradient_eps': [1e-8],
                     'weight_decay': [0, 1, 2],
                     #'lr_scheduler_step_size': [10],
                     'lr_decay': [0.5, 0.8, 1],
                     'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                     'layers': ['[512, 256, 128, 64, 32, 16, 8, 4, 2]', '[400, 200, 100, 50, 25]'],
                     'use_softmax': [False],
                     'train_percentile': [0.45, 0.5, 0.51, 0.55],
                     'display_step': [1],
                     'random_seed': [1],
                     'grid_id': ['grid_qfforma4']}


QRID_NAIVE = {'model_type': ['mean_ensemble'],
              'param' : ['soy un placeholder'],
              'grid_id': ['grid_naive']}

ALL_MODEL_SPECS  = {'mean_ensemble': {'M4': QRID_NAIVE,
                                      'M3': QRID_NAIVE,
                                      'TOURISM': QRID_NAIVE},
                    'qra': {'M4': GRID_QRA1,
                            'M3': GRID_QRA3,
                            'TOURISM': GRID_QRA3},
                    'fqra': {'M4': GRID_FQRA1,
                             'M3': GRID_FQRA1,
                             'TOURISM': GRID_FQRA1},
                    'fforma': {'M4': GRID_FFORMAM4,
                               'M3': GRID_FFORMA8,
                               'TOURISM': GRID_FFORMA1},
                    'qfforma': {'M4': GRID_QFFORMA6,
                                'M3': GRID_QFFORMA2,
                                'TOURISM': GRID_QFFORMA5}}

#############################################################################
# COMMON
#############################################################################

def generate_grid(args):
    # Declare grid directories
    grid_dir ='./results/{}/{}/'.format(args.model, args.dataset)
    grid_file_name = grid_dir + '{}_{}.csv'.format(args.model, args.dataset)

    if not os.path.exists(grid_dir):
        if not os.path.exists('./results/'):
            os.mkdir('./results/')
        if not os.path.exists('./results/{}/'.format(args.model)):
            os.mkdir('./results/{}/'.format(args.model))
        os.mkdir(grid_dir)

    # Read grid if not generate
    if not args.generate_grid:
        model_specs_df = pd.read_csv(grid_file_name)
        return model_specs_df, grid_dir

    # Generate grid
    model_specs = ALL_MODEL_SPECS[args.model][args.dataset]

    if os.path.exists(grid_dir):
        print("Erasing old files from {}, ctrl+c to cancel \n".format(grid_dir))
        time.sleep(10)
        files = glob.glob(grid_dir+'*')
        for f in files:
            os.remove(f)

    specs_list = list(itertools.product(*list(model_specs.values())))
    model_specs_df = pd.DataFrame(specs_list, columns=list(model_specs.keys()))

    model_specs_df['model_id'] = model_specs_df.index
    model_specs_df['model_id'] = model_specs_df['model_id'].astype(str)
    model_specs_df['model_id'] = model_specs_df['model_id'].str.cat(model_specs_df['grid_id'], sep='_')
    np.random.seed(1)
    model_specs_df = model_specs_df.sample(frac=1).reset_index(drop=True)

    model_specs_df.to_csv(grid_file_name, encoding='utf-8', index=None)
    return model_specs_df, grid_dir

def read_data(dataset='M4'):

    # Load and parse data
    data_file = './data/experiment/{}_pickle.p'.format(dataset)

    file = open(data_file, 'rb')
    data = pickle.load(file)

    X_train_df = data['X_train_df']
    preds_train_df = data['preds_train_df']
    y_train_df = data['y_train_df']
    y_insample_df = data['y_insample_df']

    X_test_df = data['X_test_df']
    preds_test_df = data['preds_test_df']
    y_test_df = data['y_test_df']
    fforma_errors = data.get('fforma_errors', None)

    # Filter unique_ids with X_train_df unique_ids
    unique_ids = X_train_df['unique_id'].unique()
    preds_train_df = preds_train_df[preds_train_df['unique_id'].isin(unique_ids)].reset_index(drop=True)
    y_train_df = y_train_df[y_train_df['unique_id'].isin(unique_ids)].reset_index(drop=True)
    y_insample_df = y_insample_df[y_insample_df['unique_id'].isin(unique_ids)].reset_index(drop=True)

    X_test_df = X_test_df[X_test_df['unique_id'].isin(unique_ids)].reset_index(drop=True)
    preds_test_df = preds_test_df[preds_test_df['unique_id'].isin(unique_ids)].reset_index(drop=True)
    y_test_df = y_test_df[y_test_df['unique_id'].isin(unique_ids)].reset_index(drop=True)

    # Sort datasets by unique_id, ds
    X_train_df = X_train_df.sort_values(['unique_id']).reset_index(drop=True)
    preds_train_df = preds_train_df.sort_values(['unique_id','ds']).reset_index(drop=True)
    y_train_df = y_train_df.sort_values(['unique_id','ds']).reset_index(drop=True)
    y_insample_df = y_insample_df.sort_values(['unique_id','ds']).reset_index(drop=True)

    X_test_df = X_test_df.sort_values(['unique_id']).reset_index(drop=True)
    preds_test_df = preds_test_df.sort_values(['unique_id','ds']).reset_index(drop=True)
    y_test_df = y_test_df.sort_values(['unique_id','ds']).reset_index(drop=True)
    y_test_df['ds'] = y_test_df.groupby('unique_id')['ds'].transform(lambda x: 1 + np.arange(len(x)))

    data = {'X_train_df': X_train_df,
            'preds_train_df': preds_train_df,
            'y_train_df': y_train_df,
            'y_insample_df': y_insample_df,
            'X_test_df': X_test_df,
            'preds_test_df': preds_test_df,
            'y_test_df': y_test_df,
            'fforma_errors': fforma_errors}

    return data

def train(args):
    train_model = {'mean_ensemble': train_mean_ensemble,
                   'qra': train_qra,
                   'fqra': train_fqra,
                   'fforma': train_fforma,
                   'qfforma': train_qfforma}

    # Read data
    data = read_data(args.dataset)

    # Read/Generate hyperparameter grid
    model_specs_df, grid_dir = generate_grid(args)

    # Train
    train_model[args.model](data, grid_dir, model_specs_df, args)

def upload_to_s3(args, model_id, predictions, evaluation):
    s3_dir ='{}/{}'.format(args.model, args.dataset)

    pickle_preds = f's3://research-storage-orax/{s3_dir}/predictions/{model_id}.p'
    pickle_eval = f's3://research-storage-orax/{s3_dir}/evaluation-training/{model_id}.p'

    pd.to_pickle(predictions, pickle_preds)
    pd.to_pickle(evaluation, pickle_eval)

def predict_evaluate(args, mc, model, X_test_df, preds_test_df, y_test_df):
    #output_file = '{}/model_{}.p'.format(grid_dir, mc.model_id)

    if args.model in ['qra', 'fqra', 'mean_ensemble']:
        y_hat_df = model.predict(preds_test_df)

    elif args.model in ['qfforma']:
        y_hat_df = model.predict(X_test_df, preds_test_df, y_test_df)

    elif args.model in ['fforma']:
        y_hat_df = model.predict(X_test_df, base_model_preds=preds_test_df)

    y_hat_df = y_hat_df[['unique_id', 'ds', 'y_hat']]

    evaluation_dict = {'model_id': mc.model_id,
                       'test_min_smape': model.test_min_smape,
                       'test_min_mape': model.test_min_mape}

    results_df = pd.DataFrame(evaluation_dict, index=[0])
    print(results_df)

    # Output evaluation
    if args.upload:
        mc_df = pd.DataFrame(mc.to_dict(), index=[0])
        results_df = mc_df.merge(results_df, how='left', on=['model_id'])
        upload_to_s3(args, mc.model_id, y_hat_df, results_df)
        print('Uploaded to s3!')


#############################################################################
# TRAIN CODE
#############################################################################

def train_qra(data, grid_dir, model_specs_df, args):
    # Parse data
    X_train_df = data['X_train_df']
    preds_train_df = data['preds_train_df']
    y_train_df = data['y_train_df'][['unique_id', 'ds', 'y']]
    y_insample_df = data['y_insample_df']

    X_test_df = data['X_test_df']
    preds_test_df = data['preds_test_df']
    y_test_df = data['y_test_df']

    # Parse hyper parameter data frame
    for i in range(args.start_id, args.end_id):

        mc = model_specs_df.loc[i, :]

        print(47*'=' + '\n')
        print('model_config: {}'.format(i))
        print(mc)
        print(47*'=' + '\n')

        # Instantiate, fit
        model =  LassoQuantileRegressionAveraging(tau=mc.tau, penalty=mc.penalty)
        model.fit(preds_train_df, y_train_df, preds_test_df, y_test_df[['unique_id', 'ds', 'y']])

        # Predict and Evaluate
        assert set(preds_test_df.columns) == set(preds_train_df.columns), 'columns must be the same'
        predict_evaluate(args, mc, model, X_test_df, preds_test_df, y_test_df)

def train_fqra(data, grid_dir, model_specs_df, args):
    # Parse data
    X_train_df = data['X_train_df']
    preds_train_df = data['preds_train_df']
    y_train_df = data['y_train_df'][['unique_id', 'ds', 'y']]
    y_insample_df = data['y_insample_df']

    X_test_df = data['X_test_df']
    preds_test_df = data['preds_test_df']
    y_test_df = data['y_test_df']

    # Parse hyper parameter data frame
    for i in range(args.start_id, args.end_id):

        mc = model_specs_df.loc[i, :]

        print(47*'=' + '\n')
        print('model_config: {}'.format(i))
        print(mc)
        print(47*'=' + '\n')

        # Instantiate, fit
        model = FactorQuantileRegressionAveraging(tau=mc.tau,
                                                  n_components=mc.n_components)
        model.fit(preds_train_df, y_train_df, preds_test_df, y_test_df[['unique_id', 'ds', 'y']])

        # Predict and Evaluate
        assert set(preds_test_df.columns) == set(preds_train_df.columns), 'columns must be the same'
        predict_evaluate(args, mc, model, X_test_df, preds_test_df, y_test_df)

def train_fforma(data, grid_dir, model_specs_df, args):
    # Parse data
    data_file = './data/experiment/{}_pickle.p'.format(args.dataset)
    data = pd.read_pickle(data_file)

    X_train_df = data['X_train_df']
    preds_train_df = data['preds_train_df']
    y_train_df = data['y_train_df'][['unique_id', 'ds', 'y']]
    y_insample_df = data['y_insample_df']

    X_test_df = data['X_test_df']
    preds_test_df = data['preds_test_df']
    y_test_df = data['y_test_df']
    errors = data['fforma_errors']

    from fforma.meta_learner import MetaLearnerXGBoost

    for i in range(args.start_id, args.end_id):

        mc = model_specs_df.loc[i, :]

        print(47*'=' + '\n')
        print('model_config: {}'.format(i))
        print(mc)
        print(47*'=' + '\n')

        params = {'n_estimators': int(mc.n_estimators),
                  'eta': mc.eta,
                  'max_depth': int(mc.max_depth),
                  'subsample': mc.subsample,
                  'colsample_bytree': mc.colsample_bytree,
                  'df_seasonality': None} # errors already calculated

        model = MetaLearnerXGBoost(params)

        model = model.fit(X_train_df,errors=errors, X_test_df=X_test_df,
                          preds_test_df=preds_test_df,
                          y_test_df=y_test_df[['unique_id', 'ds', 'y']])

        assert set(preds_test_df.columns) == set(preds_train_df.columns), 'columns must be the same'
        predict_evaluate(args, mc, model, X_test_df, preds_test_df, y_test_df)


def train_qfforma(data, grid_dir, model_specs_df, args):
    # Parse data
    X_train_df = data['X_train_df']
    preds_train_df = data['preds_train_df']
    y_train_df = data['y_train_df'][['unique_id', 'ds', 'y']]
    y_insample_df = data['y_insample_df']

    X_test_df = data['X_test_df']
    preds_test_df = data['preds_test_df']
    y_test_df = data['y_test_df']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    import torch
    from src.fforma import FFORMA
    from src.metrics.pytorch_metrics import WeightedPinballLoss
    from src.meta_learner import MetaLearnerNN
    #from utils import evaluate_model_prediction

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse hyper parameter data frame
    for i in range(args.start_id, args.end_id):

        mc = model_specs_df.loc[i, :]

        print(47*'=' + '\n')
        print('model_config: {}'.format(i))
        print(mc)
        print(47*'=' + '\n')

        lr_scheduler_step_size = max(mc.n_epochs // 3, 2)

        model_params = {'n_epochs': int(mc.n_epochs),
                        'lr': mc.lr,
                        'batch_size': int(mc.batch_size),
                        'gradient_eps': mc.gradient_eps,
                        'weight_decay': mc.weight_decay,
                        'lr_scheduler_step_size': int(lr_scheduler_step_size),
                        'lr_decay': mc.lr_decay,
                        'dropout': mc.dropout,
                        'layers': ast.literal_eval(mc.layers),
                        'use_softmax': mc.use_softmax,
                        'loss_function': WeightedPinballLoss(mc.train_percentile),
                        'display_step': int(mc.display_step),
                        'random_seed': int(mc.random_seed),
                        'device': device}

        # Instantiate, fit
        model = FFORMA(meta_learner_params=model_params,
                       meta_learner=MetaLearnerNN,
                       random_seed=int(mc.random_seed))
        model.fit(X_train_df, preds_train_df, y_train_df,
                  X_test_df, preds_test_df, y_test_df[['unique_id', 'ds', 'y']])

        # Predict and Evaluate
        assert set(preds_test_df.columns) == set(preds_train_df.columns), 'columns must be the same'
        predict_evaluate(args, mc, model, X_test_df, preds_test_df, y_test_df)

def train_mean_ensemble(data, grid_dir, model_specs_df, args):
    # Parse data
    X_train_df = data['X_train_df']
    preds_train_df = data['preds_train_df']
    y_train_df = data['y_train_df'][['unique_id', 'ds', 'y']]
    y_insample_df = data['y_insample_df']

    X_test_df = data['X_test_df']
    preds_test_df = data['preds_test_df']
    y_test_df = data['y_test_df']

    from benchmarks import MetaLearnerMean

    for i in range(args.start_id, args.end_id):

        mc = model_specs_df.loc[i, :]

        print(47*'=' + '\n')
        print('model_config: {}'.format(i))
        print(mc)
        print(47*'=' + '\n')

        params = {}

        model = MetaLearnerMean(params)

        model = model.fit(preds_test_df, y_test_df[['unique_id', 'ds', 'y']])

        assert set(preds_test_df.columns) == set(preds_train_df.columns), 'columns must be the same'
        predict_evaluate(args, mc, model, X_test_df, preds_test_df, y_test_df)

#############################################################################
# MAIN
#############################################################################

def parse_args():
    desc = "Experiment QFFORMA"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model', type=str, help='Model from qra, fqra, fforma, qfforma')
    parser.add_argument('--dataset', type=str, help='Daily, etc')
    parser.add_argument('--start_id', type=int, help='Start id')
    parser.add_argument('--end_id', type=int, default=0, help='End id')
    parser.add_argument('--generate_grid', type=int, default=0, help='Generate grid')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU')
    parser.add_argument('--upload', type=int, default=1, help='Upload to S3')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    assert args.dataset in ['TOURISM', 'M3', 'M4'], "Check if dataset {} is available".format(args.dataset)
    assert args.model in ALL_MODEL_SPECS.keys(), "Check if model {} is defined".format(args.model)
    train(args)
