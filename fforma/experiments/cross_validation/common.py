#!/usr/bin/env python
# coding: utf-8

from functools import partial
import joblib
from typing import Callable, Dict, Optional

import numpy as np
import optuna
from optuna import Trial
from optuna.pruners import MedianPruner
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
                 random_seed: int = 1,
                 save_study_path: Optional[str] = None) -> 'CrossValidation':
        self.meta_learner = meta_learner
        self.meta_learner_name = meta_learner.__name__.replace('MetaLearner', '')
        self.params = params
        self.default_params = default_params
        self.metric = metric
        self.metric_name = metric.__name__
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.save_study_path = save_study_path

        self.study = None

    def _fit_meta_learner(self, data: BaseData, params_trial: Dict):
        if self.meta_learner_name == 'FFNN':
            params = {**params_trial, **self.default_params}
            model = self.meta_learner(params).fit(data.features,
                                                          data.forecasts,
                                                          data.ground_truth)
        elif self.meta_learner_name == 'XGBoost':
            params = {}
            params['xgb_params'] = params_trial
            params['n_estimators'] = params['xgb_params']['n_estimators']
            params = {**params, **self.default_params}
            model = self.meta_learner(**params).fit(data.features,
                                                    data.get_metric(self.metric_name))
        else:
            raise Exception(f'Unknown meta learner: {model}')

        return model

    def _objective(self, trial: Trial, data: BaseData) -> float:

        # Data
        classes = data.features['unique_id'].str[0].values
        uids = data.features['unique_id'].values

        #kfold
        kf = StratifiedKFold(n_splits=self.n_splits,
                             shuffle=True,
                             random_state=self.random_seed)

        params_trial = self.params(trial)

        losses = []
        for step, (idx_train, idx_test) in enumerate(kf.split(uids, classes)):
            train_data = data.get_ids(uids[idx_train])
            test_data = data.get_ids(uids[idx_test])

            # Fit the model
            model = self._fit_meta_learner(train_data, params_trial)

            forecast = model.predict(test_data.features, test_data.forecasts)

            loss_test = evaluate_panel(test_data.ground_truth, forecast, self.metric)
            intermediate_value = loss_test.mean().values.item()
            losses.append(intermediate_value)

            #Pruning
            mean_intermediate_value = np.mean(losses)
            trial.report(mean_intermediate_value, step)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        losses = np.array(losses)
        mean_loss = losses.mean()
        std_loss = losses.std()

        return mean_loss

    def fit(self, data: BaseData) -> 'CrossValidation':

        objective = partial(self._objective, data=data)

        sampler = TPESampler(seed=self.random_seed)
        pruner = MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials, gc_after_trial=True)

        best_params = self.params(study.best_trial)

        self.study = study
        if self.save_study_path is not None:
            joblib.dump(study, self.save_study_path)

        self.model_ = self._fit_meta_learner(data, best_params)

        return self

    def predict(self, data: BaseData) -> pd.DataFrame:
        check_is_fitted(self, 'model_')

        forecast = self.model_.predict(data.features, data.forecasts)

        return forecast
