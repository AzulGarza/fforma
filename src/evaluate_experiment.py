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

from src.experiment import DICT_FREQS, grid_qfforma, generate_grids
from src.meta_evaluation import evaluate_fforma_experiment

def evaluate(dataset, generate, results_dir):

    # Read/Generate hyperparameter grid
    if generate:
        generate_grids(grid_dir=results_dir, model_specs=grid_qfforma)

    grid_file_name = results_dir + 'model_grid_qfforma.csv'
    model_specs_df = pd.read_csv(grid_file_name)

    errors = []
    size = len(df)
    for i, (model_id, df) in enumerate(model_specs_df.groupby('model_id')):
        print(i / size * 100, '\n)
        s3_file = f's3://research-storage-orax/{dataset}/qfforma-{model_id}.p'

        try:
            long_preds = pd.read_pickle(s3_file)
        except:
            print('File not found')
            continue

        error = evaluate_fforma_experiment(long_preds, results_dir, dataset)
        error['model_id'] = model_id
        errors.append(error)

    errors = pd.concat(errors).reset_index(drop=True)

    model_specs_df = model_specs_df.merge(errors, how='left', on=['model_id'])

    model_specs_df.to_csv(f'{results_dir}/owas_experiment_qfforma.csv')


def parse_args():
    desc = "evaluate Experiment QFFORMA"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, help='M4, M3, etc.')
    parser.add_argument('--generate_grid', type=int, default=0, help='Generate grid')

    return parser.parse_args()

def main(dataset, generate_grid):

    results_dir = './results/{}/'.format(dataset)

    print('Evaluating models...')
    evaluate(dataset, generate_grid, results_dir)

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    main(args.dataset, args.generate_grid)

# PYTHONPATH=. python -m src.evaluate_experiment --dataset 'M4' --generate_grid 1
