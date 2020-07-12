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
from sklearn.preprocessing import scale

from fforma.metrics import SMAPE1Loss
from ESRNN.utils_evaluation import owa

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

    def __init__(self, num_numerical_cols, output_size, layers, p=0.4,
                 use_softmax=False):
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

        if use_softmax:
            all_layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
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
    def __init__(self, params, val_predictions, y_df, h,
                 contribution_to_error=None,
                 random_seed=1,
                 y_train_df=None,
                 predictions_test=None,
                 y_test_df=None,
                 y_hat_benchmark='y_hat_naive2',
                 naive_seasonality=None):
        self.params = deepcopy(params)
        self.n_series, = y_df.groupby(['unique_id']).size().shape
        self.h = h
        self.n_models = val_predictions.columns.size

        actual_y = y_df['y'].values.reshape((self.n_series, self.h))
        preds_y_val = val_predictions.values.reshape((self.n_series, self.h, self.n_models))

        self.actual_y = torch.tensor(actual_y)
        self.preds_y_val = torch.tensor(preds_y_val)

        self.y_train_df = y_train_df
        self.predictions_test = predictions_test
        self.y_test_df = y_test_df
        self.y_hat_benchmark = y_hat_benchmark
        self.naive_seasonality = naive_seasonality

        self.min_owa = 4.0
        self.min_epoch = 0

        self.random_seed = random_seed

        self.loss_function = self.params.pop('loss_function', SMAPE1Loss)
        self.use_softmax = self.params.pop('use_softmax', False)

    def get_ensemble(self, margins, preds_y_val):

        weights_output = margins.repeat(1, self.h).reshape(preds_y_val.shape)

        ensemble = weights_output * preds_y_val
        ensemble = ensemble.sum(2)

        #print(ensemble)

        return ensemble

    def evaluate_model_prediction(self, features, predictions_test,
                                  y_train_df, y_test_df,
                                  y_hat_benchmark='y_hat_naive2', epoch=None):
        """
        Evaluate ESRNN model against benchmark in y_test_df
        Parameters
        ----------
        y_train_df: pandas dataframe
          panel with columns 'unique_id', 'ds', 'y'
        X_test_df: pandas dataframe
          panel with columns 'unique_id', 'ds', 'x'
        y_test_df: pandas dataframe
          panel with columns 'unique_id', 'ds', 'y' and a column
          y_hat_benchmark identifying benchmark predictions
        y_hat_benchmark: str
          column name of benchmark predictions, default y_hat_naive2

        Returns
        -------
        model_owa : float
          relative improvement of model with respect to benchmark, measured with
          the M4 overall weighted average.
        smape: float
          relative improvement of model with respect to benchmark, measured with
          the symmetric mean absolute percentage error.
        mase: float
          relative improvement of model with respect to benchmark, measured with
          the M4 mean absolute scaled error.
        """

        y_panel = y_test_df.filter(['unique_id', 'ds', 'y'])
        y_benchmark_panel = y_test_df.filter(['unique_id', 'ds', y_hat_benchmark])
        y_benchmark_panel.rename(columns={y_hat_benchmark: 'y_hat'}, inplace=True)

        #predictions
        y_hat_panel = self.predict(features)
        y_hat_panel = pd.DataFrame(y_hat_panel,
                                   index=features.index,
                                   columns=predictions_test.columns)
        y_hat_panel = (y_hat_panel * predictions_test).sum(1)
        y_hat_panel = y_hat_panel.rename('y_hat').to_frame().reset_index()

        y_insample = y_train_df.filter(['unique_id', 'ds', 'y'])

        print('y_hat:', y_hat_panel.y_hat.sum())
        print('y:', y_panel.y.sum())

        model_owa, model_mase, model_smape = owa(y_panel, y_hat_panel,
                                                 y_benchmark_panel, y_insample,
                                                 seasonality=self.naive_seasonality)

        if self.min_owa > model_owa:
          self.min_owa = model_owa
          if epoch is not None:
            self.min_epoch = epoch

        print('OWA: {} '.format(np.round(model_owa, 3)))
        print('SMAPE: {} '.format(np.round(model_smape, 3)))
        print('MASE: {} '.format(np.round(model_mase, 3)))

        return model_owa, model_mase, model_smape

    def fit(self, features, best_models=None,
            early_stopping_rounds=None, verbose_eval=True,
            features_test=None):
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
        X = scale(X)

        feats_train, \
            feats_val, \
            indices_train, \
            indices_val = train_test_split(torch.tensor(X, dtype=torch.float),
                                           torch.tensor(np.arange(X.shape[0])),
                                           random_state=self.random_seed)

        train_loss = deepcopy(self.loss_function)
        val_loss = deepcopy(self.loss_function)

        train_data = torch.utils.data.TensorDataset(feats_train, indices_train)

        if self.params['freq_of_test'] > 0:
            val_actual_y = self.actual_y[indices_val]
            val_preds_y_val = self.preds_y_val[indices_val]

        torch.manual_seed(self.random_seed)

        self.model = NeuralNetwork(num_numerical_cols=features.shape[1],
                                   output_size=self.n_models,
                                   layers=self.params['layers'],
                                   p=self.params['dropout'],
                                   use_softmax=self.use_softmax)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     betas=(0.9, 0.999),
                                     lr=self.params['learning_rate'],
                                     eps=self.params['gradient_eps'],
                                     weight_decay=self.params['weight_decay'])

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.params['batch_size'])

        for epoch in range(self.params['epochs']):

            start = time.time()
            epoch_losses = []
            for i, data in enumerate(train_loader):
                inputs, index_train = data

                train_actual_y = self.actual_y[index_train]
                train_preds_y_val = self.preds_y_val[index_train]
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                margins = self.model(inputs)
                ensemble_y_pred = self.get_ensemble(margins, train_preds_y_val)
                loss = train_loss(train_actual_y, ensemble_y_pred)

                loss.backward()

                optimizer.step()

                epoch_losses.append(loss.item())

            self.train_loss = np.mean(epoch_losses)

            if verbose_eval:
                print(f"========= Epoch {epoch} finished =========")
                print(f"Training time: {round(time.time() - start, 5)}")
                print(f"Training loss: {self.train_loss:.5f}")

            if (self.params['freq_of_test'] > 0) and (epoch % self.params['freq_of_test'] == 0):
                with torch.no_grad():
                    margins = self.model(feats_val)
                    ensemble_y_pred_test = self.get_ensemble(margins, val_preds_y_val)
                    self.test_loss = val_loss(val_actual_y, ensemble_y_pred_test)
                    print(f"Testing loss: {self.test_loss:.5f}")
                    if (features_test is not None) and (self.predictions_test is not None):
                        self.evaluate_model_prediction(features_test, self.predictions_test,
                                                       self.y_train_df, self.y_test_df,
                                                       self.y_hat_benchmark, epoch=epoch)

        return self

    def predict(self, features, tmp=1):
        """
        """
        X = features.values
        X[X != X] = 0
        X = scale(X)

        features_tensor = torch.tensor(X, dtype=torch.float)
        scores = self.model(features_tensor)

        weights = scores.detach().numpy()

        return weights
