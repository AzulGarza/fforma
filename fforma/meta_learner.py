#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb

from copy import deepcopy
from tqdm import tqdm
from scipy.special import softmax
from sklearn.model_selection import train_test_split


class MetaLearner(object):
    """

    """
    def __init__(self, params, contribution_to_error, random_seed=1):
        self.params = params
        self.contribution_to_error = contribution_to_error
        self.random_seed = random_seed

    def fobj(self, predt, dtrain):
        """
        """
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')
        preds_transformed = softmax(preds, axis=1)

        weighted_avg_loss_func = (preds_transformed * self.contribution_to_error[y, :]).sum(axis=1).reshape((n_train, 1))

        grad = preds_transformed * (self.contribution_to_error[y, :] - weighted_avg_loss_func)
        hess = self.contribution_to_error[y,:] * preds_transformed * (1.0 - preds_transformed) - grad * preds_transformed
        #hess = grad*(1 - 2*preds_transformed)
        return grad.flatten('F'), hess.flatten('F')

    def feval(self, predt, dtrain):
        """
        """
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')
        preds_transformed = softmax(preds, axis=1)
        weighted_avg_loss_func = (preds_transformed * self.contribution_to_error[y, :]).sum(axis=1)
        fforma_loss = weighted_avg_loss_func.mean()

        return 'FFORMA-loss', fforma_loss, False

    def fit(self, features, best_models, early_stopping_rounds, verbose_eval):
        """
        """
        feats_train, \
            feats_val, \
            best_models_train, \
            best_models_val, \
            indices_train, \
            indices_val = train_test_split(features,
                                           best_models,
                                           np.arange(features.shape[0]),
                                           random_state=self.random_seed,
                                           stratify=best_models)

        params = deepcopy(self.params)
        num_round = int(params.pop('n_estimators', 100))

        params['num_class'] = len(np.unique(best_models))

        dtrain = lgb.Dataset(data=feats_train, label=indices_train)
        dvalid = lgb.Dataset(data=feats_val, label=indices_val)
        valid_sets = [dtrain, dvalid]

        self.gbm_model = lgb.train(
            params=params,
            train_set=dtrain,
            fobj=self.fobj,
            num_boost_round=num_round,
            feval=self.feval,
            valid_sets=valid_sets,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )

    def predict(self, features, tmp=1):
        """
        """
        scores = self.gbm_model.predict(features, raw_score=True)
        weights = softmax(scores / tmp, axis=1)
        return weights

##############################################################################
################### CUSTOM
##############################################################################

class NeuralNetwork(nn.Module):

    def __init__(self, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        input_size = num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_numerical):
        x = self.batch_norm_num(x_numerical)
        #x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x

class RMSSETorch(nn.Module):
    def __init__(self, actual_y, preds_y_val, h, weights=None):
        super().__init__()
        self.actual_y = actual_y
        self.preds_y_val = preds_y_val
        self.h = h
        self.weights = weights
        self.softmax = nn.Softmax(1)


    def forward(self, output, labels):
        weights_output = self.softmax(output)
        weights_output = weights_output.repeat(1, self.h).reshape(self.preds_y_val.shape)
        ensemble = weights_output * self.preds_y_val
        ensemble = ensemble.sum(2)
        loss = (self.actual_y - ensemble) ** 2
        loss = loss.mean(1) ** (1 / 2) * self.weights if self.weights is not None else loss.mean(1) ** (1 / 2)
        loss = loss.sum() if self.weights is not None else loss.mean()

        return loss

class MetaLearnerNN(object):
    """

    """
    def __init__(self, actual_y, preds_y_val, h, weights=None, random_seed=1):
        #self.params = params
        self.actual_y = torch.tensor(actual_y)
        self.preds_y_val = torch.tensor(preds_y_val)
        self.h = h
        self.weights = torch.tensor(weights) if weights is not None else weights
        self.random_seed = random_seed
        self.loss_function = RMSSETorch

    def fit(self, features, best_models):
        """
        """
        feats_train, \
            feats_val, \
            best_models_train, \
            best_models_val, \
            indices_train, \
            indices_val = train_test_split(torch.tensor(features, dtype=torch.float),
                                           torch.tensor(best_models, dtype=torch.float),
                                           np.arange(features.shape[0]),
                                           random_state=self.random_seed,
                                           stratify=best_models)

        train_actual_y = self.actual_y[indices_train]
        val_actual_y = self.actual_y[indices_val]

        train_preds_y_val = self.preds_y_val[indices_train]
        val_preds_y_val = self.preds_y_val[indices_val]

        train_weights = self.weights[indices_train] if self.weights is not None else self.weights
        val_weights = self.weights[indices_val] if self.weights is not None else self.weights

        train_loss = self.loss_function(train_actual_y, train_preds_y_val,
                                        self.h, train_weights)
        val_loss = self.loss_function(val_actual_y, val_preds_y_val, self.h, val_weights)


        self.model = NeuralNetwork(features.shape[1], len(np.unique(best_models)),
                              [10, 10, 10, 10], p=0.25)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        epochs = 500
        aggregated_losses = []
        aggregated_val_losses = []

        for i in range(epochs):
            i += 1
            y_pred, y_pred_val = self.model(feats_train), self.model(feats_val)

            single_loss = train_loss(y_pred, best_models_train)
            single_loss_val = val_loss(y_pred_val, best_models_val)

            aggregated_losses.append(single_loss)
            aggregated_val_losses.append(single_loss_val)

            if i % 100 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f} val_loss: {single_loss_val.item():10.8f}')

            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f} val_loss: {single_loss_val.item():10.8f}')

        return self

    def predict(self, features, tmp=1):
        """
        """
        features_tensor = torch.tensor(features, dtype=torch.float)
        scores = self.model(features_tensor)
        weights = nn.Softmax(1)(scores / tmp).detach().numpy()
        return weights
