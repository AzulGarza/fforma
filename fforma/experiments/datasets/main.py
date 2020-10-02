#!/usr/bin/env python
# coding: utf-8

"""
Downloads datasets
------------------
- Raw datasets.
- N-BEATS forecasts.
- Pre-trained base data.
- CV-optimal parameters.
"""
import argparse
import logging

from .tourism import Tourism
from .m3 import M3


def main(directory: str) -> None:

    logger.info('\nDownloading tourism dataset')
    Tourism.download(directory)
    logger.info('\nDownloading nbeats forecasts for tourism')
    Tourism.download_nbeats_forecasts(directory)

    logger.info('\nDownloading m3 dataset')
    M3.download(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloads datasets')
    parser.add_argument('--directory', required=True, type=str,
                        help='Experiments directory')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.directory)
