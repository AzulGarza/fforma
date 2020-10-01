#!/usr/bin/env python
# coding: utf-8

import argparse
from functools import partial
import logging
from pathlib import Path
from .fnn import hpo_fnn
from .xgboost import hpo_xgboost

def main(directory: str, n_splits: int, n_trials: int, model: str):

    if model == 'ffnn':
        hpo_fnn(directory, n_splits, n_trials)
    else:
        hpo_xgboost(directory, n_splits, n_trials)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HPO for Tourism dataset')
    parser.add_argument('--directory', required=True, type=str,
                        help='experiments directory')
    parser.add_argument('--n_splits', required=True, type=int,
                        help='number of folds for kfold cv')
    parser.add_argument('--n_trials', required=True, type=int,
                        help='number of hpo trials')
    parser.add_argument('--model', required=True, type=str,
                        help='Model to cv-train (ffnn or xgboost)',
                        choices=['ffnn', 'xgboost'])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory, args.n_splits, args.n_trials, args.model)
