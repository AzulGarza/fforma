#!/usr/bin/env python
# coding: utf-8

from typing import Callable, Union
from pathlib import Path

import numpy as np
import pandas as pd

from .tourism import TourismEvaluation
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

def evaluate_dataset(directory: str, dataset: str, metric: Callable) -> pd.DataFrame:
    """Evaluates dataset.

    Parameters
    ----------
    directory: str
        Experiments directory.
    dataset: str
        Dataset to evaluate. Either 'Tourism', 'M3' or 'M4'.
    """
    assert dataset in ['Tourism'], 'Please provide either Tourism, M3 or M4'

    metric_name = metric.__name__
    path = Path(directory) / dataset.lower()

    base_data = pd.read_pickle(path / 'base' / 'base_training.p')
    benchmarks = pd.read_pickle(path / 'benchmarks' / 'benchmarks.p')

    if dataset == 'Tourism':
        base_evaluation =  _evaluate_base(base_data, TourismInfo, metric_name)
        base_evaluation = base_evaluation.loc[base_evaluation.index.str.contains('_mape')]
        benchmarks_evaluation = TourismEvaluation(directory).evaluate(benchmarks, metric)

        forecasts = []
        for group in TourismInfo.groups:
            forecast_group = pd.read_pickle(path / 'forecasts' / f'{group.name}_forecast.p')
            forecasts.append(forecast_group)
        forecasts = pd.concat(forecasts)

        forecasts_evaluation = TourismEvaluation(directory).evaluate(forecasts, metric)
        forecasts_evaluation.rename({'y_hat': 'fforma_ffnn_forec'}, inplace=True)

        evaluation = pd.concat([base_evaluation, benchmarks_evaluation, forecasts_evaluation])

        return evaluation
