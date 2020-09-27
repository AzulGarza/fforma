#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from typing import Tuple
import logging

import requests
import zipfile
import subprocess
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def data_already_present(directory: Path, kind: str, needed_data: Tuple) -> bool:
    """Checks if needed_data data is already present inside directory.

    Parameters
    ----------
    directory: Path
        Custom directory where data will be checked.
    kind: str
        Can be 'decompressed' or ....
    needed_data: Tuple
        Tuple of needed data.
    """
    if kind == 'decompressed':
        pass
    present = [os.path.exists(directory / dir) for dir in needed_data]

    present = all(present)

    return present

def maybe_download_decompress(directory: Path, source_url: str, needed_data: Tuple) -> None:
    """Download data from website, unless it's already here.

    Parameters
    ----------
    directory: Path
        Custom directory where data will be downloaded.
    source_url: str
        URL where data is hosted.
    needed_data: Tuple
        Tuple of needed data.
    """
    directory.mkdir(parents=True, exist_ok=True)

    compressed_data_directory = directory / 'raw'
    compressed_data_directory.mkdir(parents=True, exist_ok=True)

    filename = source_url.split('/')[-1]
    filepath = compressed_data_directory / filename

    if not filepath.exists():
        # Streaming, so we can iterate over the response.
        r = requests.get(source_url, stream=True)
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
            logger.error('ERROR, something went wrong downloading data')

        size = filepath.stat().st_size
        logger.info(f'Successfully downloaded {filename}, {size}, bytes.')

    if not data_already_present(directory, kind='decompressed', needed_data=needed_data):
        decompressed_data_directory = compressed_data_directory / 'decompressed_data'

        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(decompressed_data_directory)

        logger.info(f'Successfully decompressed {decompressed_data_directory}')
