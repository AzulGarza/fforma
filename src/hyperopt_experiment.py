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

from hyperopt import (
    tpe,
    STATUS_OK,
    Trials,
    hp,
    fmin,
    space_eval
)


from src.benchmarks import (
    LassoQuantileRegressionAveraging,
    FactorQuantileRegressionAveraging
)

#############################################################################
# EXPERIMENT SPECS
#############################################################################
RANDOM_STATE = 42
MAX_EVALS = 20

DICT_FREQS = {'H':24, 'D': 7, 'W':52, 'M': 12, 'Q': 4, 'Y': 1}

GRID_QFFORMA1 = {'n_epochs' : hp.quniform('n_epochs', 5, 50, 1),
                 'lr': hp.loguniform('lr', -5.0, -2.3),
                 'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 200, 256, 512]),
                 'weight_decay':  hp.loguniform('weight_decay', -5.0, -2.3),
                 'lr_scheduler_step_size': hp.quniform('lr_scheduler_step_size', 2, 20, 1),
                 'lr_decay': hp.uniform('lr_decay', 0.01, 1.0),
                 'dropout': hp.uniform('dropput', 0., 1),
                 #'use_softmax': hp.choice('use_softmax', [True, False]),
                 'train_percentile': hp.uniform('train_percentile', 0.4, 0.7)}

ALL_MODEL_SPECS  = {'qfforma': {'M4': GRID_QFFORMA1,
                                'TOURISM': GRID_QFFORMA1}}

#############################################################################
# COMMON
#############################################################################

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
    train_model = {'qfforma': train_qfforma}

    # Read data
    data = read_data(args.dataset)

    # Train
    train_model[args.model](data, args)


#############################################################################
# TRAIN CODE
#############################################################################

def train_qfforma(data, args):
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    space = ALL_MODEL_SPECS['qfforma'][args.dataset]
    tpe_algorithm = tpe.suggest
    # Parse hyper parameter data frame
    print(47*'=' + '\n')
    print('Initializing optmization process')
    print(47*'=' + '\n')

    FIXED_PARAMS = {'device': device,
                    'gradient_eps': 1e-8,
                    'layers': [512, 256, 128, 64, 32, 16, 8, 4, 2],
                    'random_seed': RANDOM_STATE,
                    'use_softmax': True}

    def objective(params):

        model_params = {**FIXED_PARAMS, **params}
        model_params['loss_function'] = WeightedPinballLoss(model_params['train_percentile'])

        model_params['batch_size'] = int(model_params['batch_size'])
        model_params['n_epochs'] = int(model_params['n_epochs'])

        model_params['display_step'] = model_params['n_epochs']
        # Instantiate, fit
        model = FFORMA(meta_learner_params=model_params,
                       meta_learner=MetaLearnerNN,
                       random_seed=RANDOM_STATE)
        model.fit(X_train_df, preds_train_df, y_train_df,
                  X_test_df, preds_test_df, y_test_df[['unique_id', 'ds', 'y']],
                  verbose=False)

        return {'loss': model.test_min_smape, 'params': params, 'status': STATUS_OK}

    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=MAX_EVALS,
                rstate=np.random.RandomState(RANDOM_STATE))

    optimal_params = {**FIXED_PARAMS, **space_eval(space, best)}

    pd.to_pickle(optimal_params, 'opt-params.p')

    print(optimal_params)



    # Predict and Evaluate
    #assert set(preds_test_df.columns) == set(preds_train_df.columns), 'columns must be the same'
    #predict_evaluate(args, mc, model, X_test_df, preds_test_df, y_test_df)

#############################################################################
# MAIN
#############################################################################

def parse_args():
    desc = "Experiment QFFORMA"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model', type=str, help='Model from qra, fqra, fforma, qfforma')
    parser.add_argument('--dataset', type=str, help='Daily, etc')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    assert args.dataset in ['TOURISM', 'M3', 'M4'], "Check if dataset {} is available".format(args.dataset)
    assert args.model in ALL_MODEL_SPECS.keys(), "Check if model {} is defined".format(args.model)
    train(args)
