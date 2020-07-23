import os
import yaml
import argparse
import itertools
import ast
import pickle
import time

import numpy as np
import pandas as pd
import glob

DICT_FREQS = {'H':24, 'D': 7, 'W':52, 'M': 12, 'Q': 4, 'Y': 1}

grid_qfforma = {'model_type': ['qfforma'],
                'n_epochs' : [5, 20, 50],
                'lr': [1e-4, 1e-3, 1e-2],
                'batch_size': [128],
                'gradient_eps': [1e-8],
                'weight_decay': [0],
                'lr_scheduler_step_size': [5],
                'lr_decay': [0.4],
                'dropout': [0.1],
                'layers': [[200, 100, 50, 25, 10]],
                'use_softmax': [False, True],
                'train_percentile': [0.4, 0.5, 0.6],
                'random_seed': [1]}

def generate_grids(grid_dir, model_specs):

    if not os.path.exists(grid_dir):
        if not os.path.exists('./results/'):
            os.mkdir('./results/')
        os.mkdir(grid_dir)

    if os.path.exists(grid_dir):
        print("Erasing old files from {}, ctrl+c to cancel \n".format(grid_dir))
        time.sleep(10)
        files = glob.glob(grid_dir+'*')
        for f in files:
            os.remove(f)

    # Read/Generate hyperparameter grid
    specs_list = list(itertools.product(*list(model_specs.values())))
    model_specs_df = pd.DataFrame(specs_list, columns=list(model_specs.keys()))

    model_specs_df['model_id'] = model_specs_df.index
    np.random.seed(1)
    model_specs_df = model_specs_df.sample(frac=1).reset_index(drop=True)

    grid_file_name = grid_dir + 'model_grid_qfforma.csv'
    model_specs_df.to_csv(grid_file_name, encoding='utf-8', index=None)

def uploat_to_s3(mc_row, evaluation_dict, dataset):

    mc_dict = mc_row.to_dict()
    data = {**mc_dict, **evaluation_dict}
    data = pd.DataFrame(data, index=[0])

    pickle_file = f's3://research-storage-orax/{dataset}/qfforma-{mc.model_id}.p'
    pd.to_pickle(data, pickle_file)


def train_qfforma(data, start_id, end_id, dataset, generate, gpu_id=0):

    # Read/Generate hyperparameter grid
    if generate:
        generate_grids(grid_dir=results_dir, model_specs=grid_qfforma)

    grid_file_name = results_dir + 'model_grid_qfforma.csv'
    model_specs_df = pd.read_csv(grid_file_name)

    # Parse data
    X_train_df = data['X_train_df']
    preds_train_df = data['preds_train_df']
    y_train_df = data['y_train_df']
    y_insample_df = data['y_insample_df']

    X_test_df = data['X_test_df']
    preds_test_df = data['preds_test_df']
    y_test_df = data['y_test_df']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    from fforma import FFORMA
    from metrics.pytorch_metrics import WeightedPinballLoss
    from meta_learner import MetaLearnerNN
    from utils import evaluate_model_prediction

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'

    # Parse hyper parameter data frame
    for i in range(start_id, end_id):

        mc = model_specs_df.loc[i, :]

        print(47*'=' + '\n')
        print('model_config: {}'.format(i))
        print(mc)
        print(47*'=' + '\n')

        # Check if result already exists
        #output_file = '{}/model_{}.p'.format(results_dir, mc.model_id)

        model_params = {'n_epochs': int(mc.n_epochs),
                        'lr': mc.lr,
                        'batch_size': int(mc.batch_size),
                        'gradient_eps': mc.gradient_eps,
                        'weight_decay': mc.weight_decay,
                        'lr_scheduler_step_size': int(mc.lr_scheduler_step_size),
                        'lr_decay': mc.lr_decay,
                        'dropout': mc.dropout,
                        'layers': ast.literal_eval(mc.layers),
                        'use_softmax': mc.use_softmax,
                        'loss_function': WeightedPinballLoss(mc.train_percentile),
                        'random_seed': int(mc.random_seed),
                        'device': device}

        # Instantiate, fit, predict and evaluate
        model = FFORMA(meta_learner_params=model_params,
                       meta_learner=MetaLearnerNN,
                       random_seed=int(mc.random_seed))
        model.fit(X_train_df, preds_train_df, y_train_df)
        y_hat_df = model.predict(X_test_df, preds_test_df, y_test_df)

        # Infer seasonalities
        seasonalities = y_test_df.groupby('unique_id')['ds'].apply(lambda x: pd.infer_freq(x))
        seasonalities = seasonalities.rename('freq').reset_index()
        seasonalities['seasonality'] = seasonalities['freq'].replace(DICT_FREQS)
        assert seasonalities['seasonality'].isnull().sum() == 0
        seasonalities = seasonalities.set_index('unique_id')
        seasonalities = seasonalities.to_dict()['seasonality']

        model_owa, model_mase, model_smape = evaluate_model_prediction(y_train_df=y_insample_df,
                                                                       outputs_df=y_hat_df,
                                                                       seasonalities=seasonalities)

        print("OWA: {:03.3f}".format(model_owa))
        print("MASE: {:03.3f}".format(model_mase))
        print("SMAPE: {:03.3f}".format(model_smape))

        evaluation_dict = {'id': mc.model_id,
                          'owa': model_owa,
                          'mase': model_mase,
                          'smape': model_smape}

        # Output evaluation
        uploat_to_s3(mc, evaluation_dict, dataset)

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

    data = {'X_train_df': X_train_df,
            'preds_train_df': preds_train_df,
            'y_train_df': y_train_df,
            'y_insample_df': y_insample_df,
            'X_test_df': X_test_df,
            'preds_test_df': preds_test_df,
            'y_test_df': y_test_df}

    return data

def parse_args():
    desc = "Experiment QFFORMA"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, help='Daily, etc')
    parser.add_argument('--start_id', type=int, help='Start id')
    parser.add_argument('--end_id', type=int, default=0, help='End id')
    parser.add_argument('--generate_grid', type=int, default=0, help='Generate grid')
    parser.add_argument('--gpu_id', type=int, help='GPU')
    return parser.parse_args()

def main(dataset, start_id, end_id, generate_grid, gpu_id):

    #results_dir = './results/{}/'.format(dataset)
    print("Reading data...")
    data = read_data(dataset)

    print('Training model...')
    train_qfforma(data, start_id, end_id, dataset, generate_grid, gpu_id)

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    main(args.dataset, args.start_id, args.end_id, args.generate_grid, args.gpu_id)

# PYTHONPATH=. python src/experiment.py --dataset 'M4' --start_id 1 --end_id 2 --generate_grid 0 --gpu_id 3
