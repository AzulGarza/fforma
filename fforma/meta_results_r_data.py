#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import requests
import os
import tarfile
import subprocess
import pandas as pd

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
    filename: str
        Filename of M4 data with format /Type/Frequency.csv. Example: /Test/Daily-train.csv
    directory: str
        Custom directory where data will be downloaded.
    """
    compressed_data_directory = directory + "/raw"

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

    decompressed_data_directory = compressed_data_directory + '/decompressed_data'

    tar = tarfile.open(filepath, 'r:gz')
    tar.extractall(path=decompressed_data_directory)
    tar.close()

    print('Successfully decompressed ', decompressed_data_directory)

def prepare_fforma_data(directory, dataset_name=None):

    #Check downloaded data
    maybe_download_decompress(directory)

    #Prepare data from R
    cmd = f'Rscript ./fforma/R/prepare_data_m4.R "{directory}"'
    res_r = os.system(cmd)

    assert res_r == 0, 'Some error happened with R processing'

    feats_train = pd.read_csv(directory + '/processed_data/train-features.csv')
    X_models_train = pd.read_csv(directory + '/processed_data/train-ff.csv')
    y_models_train = pd.read_csv(directory + '/processed_data/train-xx.csv')

    feats_test = pd.read_csv(directory + '/processed_data/test-features.csv')
    X_models_test = pd.read_csv(directory + '/processed_data/test-ff.csv')

    if dataset_name is not None:
        kind = dataset_name[0]

        feats_train = feats_train[feats_train['unique_id'].str.startswith(kind)]
        X_models_train = X_models_train[X_models_train['unique_id'].str.startswith(kind)]
        y_models_train = y_models_train[y_models_train['unique_id'].str.startswith(kind)]

        feats_test = feats_test[feats_test['unique_id'].str.startswith(kind)]
        X_models_test = X_models_test[X_models_test['unique_id'].str.startswith(kind)]


    return feats_train, X_models_train, y_models_train, feats_test, X_models_test
