#!/usr/bin/env python
# coding: utf-8

from typing import Callable

import numpy as np
import pandas as pd

from fforma.experiments.datasets.tourism import Tourism, TourismInfo
from fforma.utils.evaluation import evaluate_models


class TourismEvaluation:

    def __init__(self, directory: str) -> None:
        self.test_set = Tourism.load(directory, training=False)

    def evaluate(self, forecast: pd.DataFrame, metric: Callable) -> pd.DataFrame:

        losses = []
        obs = []
        for group in TourismInfo.groups:
            name = group.name

            test_group = self.test_set.get_group(name)
            ids = test_group.groups[name]
            forecasts_group = forecast.query('unique_id in @ids')

            loss_group = evaluate_models(test_group.y, forecasts_group, metric)
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
