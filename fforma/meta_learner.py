#!/usr/bin/env python
# coding: utf-8

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb

from copy import deepcopy
from tqdm import tqdm
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from fforma.utils.losses import RMSSETorch

softmax = nn.Softmax(1)


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

class MetaLearnerNN(object):
    """Evaluates ensemble model on the fly using neural networks.

    Parameters
    ----------
    actual_y: numpy array
        Actual values of the time series.
        Numpy array of size N * h
    preds_y_val: numpy array
        Model predictions to ensemble.
        Numpy array of size N * h * m.
    h: int
        Horizon of the validation set.
    weights: numpy array
        Weighted errors.
    loss_function: pytorch loss function

    random_seed:

    """
    def __init__(self, params, y_df, val_predictions, h,
                 contribution_to_error=None,
                 random_seed=1):
        self.params = deepcopy(params)
        self.n_series, = y_df['unique_id'].unique().shape
        self.h = h
        self.n_models, = val_predictions.drop(['unique_id', 'ds', 'y'], 1).columns.shape

        actual_y = y_df['y'].values.reshape((self.n_series, self.h))
        preds_y_val = val_predictions.drop(columns=['unique_id', 'ds', 'y'])
        preds_y_val = preds_y_val.values.reshape((self.n_series, self.h, self.n_models))

        self.actual_y = torch.tensor(actual_y)
        self.preds_y_val = torch.tensor(preds_y_val)

        self.random_seed = random_seed

        self.loss_function = self.params.pop('loss_function', RMSSETorch)
        self.use_softmax = self.params.pop('use_softmax', False)

    def get_ensemble(self, margins, preds_y_val):
        if self.use_softmax:
            weights_output = softmax(margins)
        else:
            weights_output = margins

        weights_output = weights_output.repeat(1, self.h).reshape(preds_y_val.shape)

        ensemble = weights_output * preds_y_val
        ensemble = ensemble.sum(2)

        #print(ensemble)

        return ensemble

    def fit(self, features, best_models, early_stopping_rounds, verbose_eval=True):
        """
        Parameters
        ----------
        features: numpy array
            Numpy array of size N * f.
        best_models: numpy array
            Numpy array of size N.
        """
        X = features.values
        X[X != X] = 0
        X = (X - X.mean()) / X.std()



        feats_train, \
            feats_val, \
            best_models_train, \
            best_models_val, \
            indices_train, \
            indices_val = train_test_split(torch.tensor(X, dtype=torch.float),
                                           torch.tensor(best_models, dtype=torch.float),
                                           torch.tensor(np.arange(X.shape[0])),
                                           random_state=self.random_seed,
                                           stratify=best_models)

        train_loss = deepcopy(self.loss_function)
        val_loss = deepcopy(self.loss_function)

        train_data = torch.utils.data.TensorDataset(feats_train, best_models_train,
                                                    indices_train)

        if self.params['freq_of_test'] > 0:
            val_actual_y = self.actual_y[indices_val]
            val_preds_y_val = self.preds_y_val[indices_val]
            #val_weights = self.weights[indices_val] if self.weights is not None else self.weights

        self.model = NeuralNetwork(num_numerical_cols=features.shape[1],
                                   output_size=self.n_models,
                                   layers=self.params['layers'],
                                   p=self.params['dropout'])

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     betas=(0.9, 0.999),
                                     lr=self.params['learning_rate'],
                                     eps=self.params['gradient_eps'],
                                     weight_decay=self.params['weight_decay'])

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.params['batch_size'])
        #aggregated_losses = []
        #aggregated_val_losses = []

        for epoch in range(self.params['epochs']):

            start = time.time()
            epoch_losses = []
            for i, data in enumerate(train_loader):
                inputs, labels, index_train = data

                train_actual_y = self.actual_y[index_train]
                #val_actual_y = self.actual_y[indices_val]
                train_preds_y_val = self.preds_y_val[index_train]
                #val_preds_y_val = self.preds_y_val[indices_val]
                #train_weights = self.weights[index_train] if self.weights is not None else self.weights
                #val_weights = self.weights[indices_val] if self.weights is not None else self.weights
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                margins = self.model(inputs)

                ensemble_y_pred = self.get_ensemble(margins, train_preds_y_val)
                loss = train_loss(train_actual_y, ensemble_y_pred)

                loss.backward()
                optimizer.step()
                #loss_val = val_loss(y_pred_val, best_models_val)


                epoch_losses.append(loss.item())
                #aggregated_losses.append(single_loss)
                #aggregated_val_losses.append(single_loss_val)

                # print statistics
                #if i % 50 == 1:
                #    print(f'[{epoch + 1: 3}, {i + 1: 3}] loss: {np.mean(epoch_losses): 10.8f}')

            self.train_loss = np.mean(epoch_losses) # if self.weights is not None else np.mean(epoch_losses)

            if verbose_eval:
                print(f"========= Epoch {epoch} finished =========")
                print(f"Training time: {round(time.time() - start, 5)}")
                print(f"Training loss: {self.train_loss:.5f}")

            if (self.params['freq_of_test'] > 0) and (epoch % self.params['freq_of_test'] == 0):
                margins = self.model(feats_val)
                ensemble_y_pred_test = self.get_ensemble(margins, val_preds_y_val)
                self.test_loss = val_loss(val_actual_y, ensemble_y_pred_test)
                print(f"Testing loss: {self.test_loss:.5f}")

        return self

    def predict(self, features, tmp=1):
        """
        """
        X = features.values
        X[X != X] = 0
        X = (X - X.mean()) / X.std()

        features_tensor = torch.tensor(X, dtype=torch.float)
        scores = self.model(features_tensor)

        if self.use_softmax:
            weights = nn.Softmax(1)(scores / tmp).detach().numpy()
        else:
            weights = scores.detach().numpy()

        return weights
