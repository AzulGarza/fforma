#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from fforma.base import FQRA, QRAL1


class MetaLearnerFQRA(object):

    def __init__(self, tau, n_components, scheduler='processes'):
        """
        """
        self.tau = tau
        self.n_components = n_components
        self.scheduler = scheduler
        self.model = {'FQRA': FQRA(n_components=self.n_components, tau=self.tau)}

    def fit(self, X_df, y_df, X_test_df, y_test_df):
        """
        """
        train_df = train_to_horizontal(X_df, y_df)
        print('Widing finished.')
        train_df['seasonality']= 12 #TODO: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXoXXXXXXXXXXXXX
        test_df = train_to_horizontal(X_test_df, y_test_df)
        print('Widing finished.')

        self.meta_model = MetaModels(models=self.model, scheduler=self.scheduler)
        self.meta_model.fit(y_panel_df=train_df)

        y_hat_ = self.meta_model.predict(test_df)
        y_hat_ = y_hat_[['unique_id','ds','FQRA']]
        y_hat_.columns = ['unique_id', 'ds', 'y_hat']

        y_hat_ = wide_to_long(y_hat_, lst_cols=['y_hat', 'ds'])

        self.y_hat_ = y_hat_

        return self

    def predict(self, X_df):
        """
        """
        check_is_fitted(self)

        return self.y_hat_

class MetaLearnerLQRA(object):

    def __init__(self, tau, penalty, scheduler='processes'):
        self.tau = tau
        self.penalty = penalty
        self.scheduler = scheduler
        self.model = {'LQRA': QRAL1(tau=self.tau, lambd=self.penalty)}

    def fit(self, X_df, y_df, X_test_df, y_test_df):
        """
        """
        train_df = train_to_horizontal(X_df, y_df)
        train_df['seasonality']= 12 #TODO: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXoXXXXXXXXXXXXX
        test_df = train_to_horizontal(X_test_df, y_test_df)

        self.meta_model = MetaModels(models=self.model, scheduler=self.scheduler)
        self.meta_model.fit(y_panel_df=train_df)
        self.meta_model.scheduler = 'single-threaded'

        y_hat_df = self.meta_model.predict(test_df)
        y_hat_df = y_hat_df[['unique_id','ds','LQRA']]
        y_hat_df.columns = ['unique_id', 'ds', 'y_hat']

        self.y_hat_df = y_hat_df

        y_hat_df = wide_to_long(y_hat_df, lst_cols=['y_hat','ds'])

        self.y_hat_df = y_hat_df

        self.test_min_smape = evaluate_panel(y_panel=y_test_df,
                                             y_hat_panel=y_hat_df,
                                             metric=smape)
        self.test_min_mape = evaluate_panel(y_panel=y_test_df,
                                            y_hat_panel=y_hat_df,
                                            metric=mape)

        return self

    def predict(self, X_df):
        """
        """
        y_hat = self.y_hat_df

        return y_hat
