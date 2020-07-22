#!/usr/bin/env python
# coding: utf-8

import requests
import os
import tarfile
import subprocess

import pandas as pd

from tqdm import tqdm
from src.m4_data import prepare_m4_data, prepare_full_m4_data

URL = 'https://github.com/pmontman/M4metaresults/releases/download/v0.0.0.9000/M4metaresults_0.0.0.9000.tar.gz'

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def maybe_download_decompress(directory):
    """Download the data from M4's website, unless it's already here.

    Parameters
    ----------
    directory: str
        Custom directory where data will be downloaded.
    """
    root_fforma_data = directory + '/hyndman_data'
    if not os.path.exists(root_fforma_data):
        os.mkdir(root_fforma_data)

    compressed_data_directory = root_fforma_data + '/raw'

    if not os.path.exists(compressed_data_directory):
        os.mkdir(compressed_data_directory)

    filename = URL.split('/')[-1]
    filepath = os.path.join(compressed_data_directory, filename)

    if not os.path.exists(filepath):
        # Streaming, so we can iterate over the response.
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

    if not data_already_present(directory, kind='decompressed'):
        decompressed_data_directory = compressed_data_directory + '/decompressed_data'

        tar = tarfile.open(filepath, 'r:gz')
        tar.extractall(path=decompressed_data_directory)
        tar.close()

        print('Successfully decompressed ', decompressed_data_directory)

def data_already_present(directory, kind):
    """
    kind: str
        Can be 'r' o 'decompressed'.
    """
    if kind == 'r':
        needed_data = ('/hyndman_data/processed_data/train-features.csv',
                       '/hyndman_data/processed_data/train-ff.csv',
                       '/hyndman_data/processed_data/train-xx.csv',
                       '/hyndman_data/processed_data/test-features.csv',
                       '/hyndman_data/processed_data/test-ff.csv')
    elif kind == 'decompressed':
        main_dir = '/hyndman_data/raw/decompressed_data/M4metaresults/data'
        needed_data = (f'{main_dir}/submission_M4.rda',
                       f'{main_dir}/meta_M4.rda',
                       f'{main_dir}/model_M4.rda')

    present = [os.path.exists(directory + dir) for dir in needed_data]

    present = all(present)

    return present

def prepare_fforma_data(directory, dataset_name=None):

    #Check downloaded data
    maybe_download_decompress(directory)

    # #Prepare data from R
    # if not data_already_present(directory, kind='r'):
    #     cmd = f'Rscript ./fforma/R/prepare_data_m4.R "{directory}"'
    #     res_r = os.system(cmd)

    #     assert res_r == 0, 'Some error happened with R processing'

    root_processed_data = directory + '/hyndman_data/processed_data'

    X_train_df = pd.read_csv(root_processed_data + '/train-features.csv')
    preds_train_df = pd.read_csv(root_processed_data + '/train-ff.csv')
    y_train_df = pd.read_csv(root_processed_data + '/train-xx.csv')

    X_test_df = pd.read_csv(root_processed_data + '/test-features.csv')
    preds_test_df = pd.read_csv(root_processed_data + '/test-ff.csv')


    if dataset_name is not None:
        kind = dataset_name[0]

        X_train_df = X_train_df[X_train_df['unique_id'].str.startswith(kind)]
        preds_train_df = preds_train_df[preds_train_df['unique_id'].str.startswith(kind)]
        y_train_df = y_train_df[y_train_df['unique_id'].str.startswith(kind)]

        X_test_df = X_test_df[X_test_df['unique_id'].str.startswith(kind)]
        preds_test_df = preds_test_df[preds_test_df['unique_id'].str.startswith(kind)]
        _, y_insample_df, _, y_test_df = prepare_m4_data(dataset_name, directory, 100_000)
    else:
        _, y_insample_df, _, y_test_df = prepare_full_m4_data(directory)

    return X_train_df, preds_train_df, y_train_df, X_test_df, preds_test_df, y_insample_df, y_test_df
