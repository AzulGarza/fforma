#!/usr/bin/env python
# coding: utf-8

from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd

from fforma.utils.evaluation import evaluate_models
from fforma.experiments.datasets.tourism import TourismInfo
from fforma.experiments.base.common import BaseData


def _evaluate_base(base_data: BaseData,
                   info: Union[TourismInfo],
                   metric: str) -> pd.DataFrame:
    """Evaluates base data."""
    losses = []
    obs = []
    for group in info.groups:
        name = group.name

        loss_group = base_data.get_group(name).get_metric(metric)
        obs_group = loss_group.shape[0] * group.horizon

        loss_group = loss_group.mean().rename(name).to_frame()

        losses.append(loss_group)
        obs.append(obs_group)

    losses = pd.concat(losses, 1)
    obs = np.array(obs)

    total = (losses * obs).sum(1) / obs.sum()
    total = total.rename('Average').to_frame()
    losses = losses.join(total)

    return losses

def evaluate_dataset(directory: str, dataset: str, metric: str = 'mape') -> pd.DataFrame:
    """Evaluates dataset.

    Parameters
    ----------
    directory: str
        Experiments directory.
    dataset: str
        Dataset to evaluate. Either 'Tourism', 'M3' or 'M4'.
    """
    assert dataset in ['Tourism'], 'Please provide either Tourism, M3 or M4'

    path = Path(directory) / dataset.lower() / 'base' / 'base_training.p'

    base_data = pd.read_pickle(path)

    if dataset == 'Tourism':
        return _evaluate_base(base_data, TourismInfo, metric)
