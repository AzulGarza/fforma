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

def uploat_evaluation_to_s3(model_id, data, dataset):

    #mc_dict = mc_row.to_dict()
    #data = {**mc_dict, **evaluation_dict}
    #data = pd.DataFrame(data, index=[0])

    pickle_file = f's3://research-storage-orax/{dataset}/evaluation/qfforma-evaluation-{model_id}.p'
    pd.to_pickle(data, pickle_file)

def evaluate(dataset, start_id, end_id, generate, results_dir):

    # Read/Generate hyperparameter grid
    if generate:
        generate_grids(grid_dir=results_dir, model_specs=grid_qfforma)

    grid_file_name = results_dir + 'model_grid_qfforma.csv'
    model_specs_df = pd.read_csv(grid_file_name)

    size = end_id - start_id
    for idx, i in enumerate(range(start_id, end_id)):
        mc = model_specs_df.loc[[i], :]
        model_id = mc['model_id'].item()

        s3_file = f's3://research-storage-orax/{dataset}/qfforma-{model_id}.p'

        try:
            long_preds = pd.read_pickle(s3_file)
            print(idx / size * 100, '\n')
            print('File found!', 'preparing to evaluate')
        except:
            continue

        error = evaluate_fforma_experiment(long_preds, results_dir, dataset)
        error['model_id'] = model_id

        error = mc.merge(error, how='left', on=['model_id'])

        uploat_evaluation_to_s3(model_id, error, dataset)

def parse_args():
    desc = "evaluate Experiment QFFORMA"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, help='M4, M3, etc.')
    parser.add_argument('--generate_grid', type=int, default=0, help='Generate grid')
    parser.add_argument('--start_id', type=int, help='Start id')
    parser.add_argument('--end_id', type=int, default=0, help='End id')

    return parser.parse_args()

def main(dataset, start_id, end_id, generate_grid):

    results_dir = './results/{}/'.format(dataset)

    print('Evaluating models...')
    evaluate(dataset, start_id, end_id, generate_grid, results_dir)

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    main(args.dataset, args.start_id, args.end_id, args.generate_grid)

# PYTHONPATH=. python -m src.evaluate_experiment --dataset 'M4' --start_id 1 --end_id 2 --generate_grid 1
