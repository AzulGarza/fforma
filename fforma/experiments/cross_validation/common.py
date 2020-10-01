#!/usr/bin/env python
# coding: utf-8

from functools import partial
from typing import Callable, Dict, Optional

import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import StratifiedKFold

from fforma.experiments.base.common import BaseData
from fforma.utils.evaluation import evaluate_panel


class CrossValidation:

    def __init__(self, meta_learner, params: Callable, default_params: Optional[Dict] = {},
                 metric = Callable,
                 n_splits: int = 5, n_trials: int = 100,
                 random_seed: int = 1) -> 'CrossValidation':
        self.meta_learner = meta_learner
        self.meta_learner_name = meta_learner.__name__.replace('MetaLearner', '')
        self.params = params
        self.default_params = default_params
        self.metric = metric
        self.metric_name = metric.__name__
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_seed = random_seed

        self.study = None

    def _fit_meta_learner(self, data: BaseData, params_trial: Dict):
        if self.meta_learner_name == 'FFNN':
            model = self.meta_learner(params_trial).fit(data.features,
                                                        data.forecasts,
                                                        data.ground_truth)
        else:
            model = self.meta_learner(params_trial).fit(data.features,
                                                        data.get_metric(self.metric_name))

        return model

    def _objective(self, trial: Trial, data: BaseData) -> float:

        # Data
        classes = data.features['unique_id'].str[0].values
        uids = data.features['unique_id'].values

        #kfold
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)

        params_trial = {**self.params(trial), **self.default_params}

        losses = []
        for idx_train, idx_test in kf.split(uids, classes):
            train_data = data.get_ids(uids[idx_train])
            test_data = data.get_ids(uids[idx_test])

            # Fit the model
            model = self._fit_meta_learner(train_data, params_trial)

            forecast = model.predict(test_data.features, test_data.forecasts)

            loss_test = evaluate_panel(test_data.ground_truth, forecast, self.metric)

            losses.append(loss_test.mean().values.item())

        losses = np.array(losses)
        mean_loss = losses.mean()
        std_loss = losses.std()

        return mean_loss

    def fit(self, data: BaseData) -> 'CrossValidation':

        objective = partial(self._objective, data=data)

        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, gc_after_trial=True)

        best_params = {**self.params(study.best_trial), **self.default_params}

        self.study = study
        self.model_ = self._fit_meta_learner(data, best_params)

        return self

    def predict(self, data: BaseData) -> pd.DataFrame:
        check_is_fitted(self, 'model_')

        forecast = self.model_.predict(data.features, data.forecasts)

        return forecast
