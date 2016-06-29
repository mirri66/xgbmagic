from __future__ import absolute_import, division, print_function, unicode_literals
from xgboost.sklearn import XGBClassifier, XGBRegressor
import xgboost as xgb
import pandas as pd
import operator
import seaborn as sns
import numpy as np
from sklearn import grid_search, metrics
import unicodecsv as csv
from sklearn.externals import joblib
import random, copy

class Xgb:
    def __init__(self, df, target_column='', id_column='', target_type='binary', categorical_columns=[], drop_columns=[], numeric_columns=[], num_training_rounds=500, verbose=1, sample_fraction=1.0, n_samples=1, early_stopping_rounds=None, prefix='xgb_model', scoring=None):
        """
        input params:
        - df (DataFrame): dataframe of training data
        - target_column (string): name of target column
        - id_column (string): name of id column
        - target_type (string): 'linear' or 'binary'
        - categorical_columns (list): list of column names of categorical data. Will perform one-hot encoding
        - drop_columns (list): list of columns to drop
        - numeric_columns (list): list of columns to convert to numeric
        - verbose (bool): verbosity of printouts
        """
        # checks for sampling
        sample_fraction = float(sample_fraction)
        if sample_fraction > 1:
            sample_fraction = 1.0
        if sample_fraction * n_samples > 1:
            n_samples = round(1.0/sample_fraction)
        if sample_fraction <= 0:
            print('sample_fraction 0 or negative, switching to 0.1')
            sample_fraction = 0.1
        # if sample_fraction is results in sample smaller than 1
        if round(sample_fraction * len(df)) == 0:
            sample_fraction = 1.0/len(df)
        # check if data is dataframe
        if type(df) == pd.core.frame.DataFrame:
            self.df = df
            self.early_stopping_rounds = early_stopping_rounds
            if target_column:
                self.target_column = target_column
                self.id_column = id_column
                self.target_type = target_type
                self.categorical_columns = categorical_columns
                self.numeric_columns = numeric_columns
                self.drop_columns = drop_columns
                self.verbose = verbose
                self.sample_fraction = sample_fraction
                self.n_samples = n_samples
                self.num_training_rounds = num_training_rounds
                self.prefix = prefix
                # init the classifier:
                if self.target_type == 'binary' or self.target_type == 'multiclass':
                    self.scoring = 'auc'
                    self.clf = XGBClassifier(
                        learning_rate =0.1,
                        n_estimators = num_training_rounds,
                        subsample = 0.8,
                        colsample_bytree = 0.8,
                        objective = 'binary:logistic',
                        scale_pos_weight = 1,
                        seed = 123)
                elif self.target_type == 'linear':
                    self.scoring = 'rmse'
                    self.clf = XGBRegressor(
                            n_estimators = num_training_rounds,
                            objective = 'reg:linear'
                            )
                # if preferred scoring metric is stated:
                if scoring:
                    self.scoring = scoring
            else:
                print('please provide target column name')
        else:
            print('please provide pandas dataframe')

    def train(self):
        print('#### preprocessing ####')
        self.df = self.preprocess(self.df)

        print('#### training ####')
        self.predictors = [x for x in self.df.columns if x not in [self.target_column, self.id_column]]
        xgb_param = self.clf.get_xgb_params()

        # if subsampling
        if self.sample_fraction == 1.0:
            df_list = [self.df]
        else:
            df_list = self.random_sample(df=self.df, fraction=self.sample_fraction, n_samples=self.n_samples)
        print(df_list)
        for idx, current_df in enumerate(df_list):
            print('ITERATION ' + str(idx) + ' of ' + str(self.n_samples) +', sample_fraction=' + str(self.sample_fraction))
            xgtrain  = xgb.DMatrix(current_df[self.predictors], label=current_df[self.target_column], missing=np.nan)
            try:
                cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=self.clf.get_params()['n_estimators'], nfold=5,
                    metrics=[self.scoring], early_stopping_rounds=self.early_stopping_rounds, show_progress=self.verbose)
            except:
                try:
                    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=self.clf.get_params()['n_estimators'], nfold=5,
                        metrics=[self.scoring], early_stopping_rounds=self.early_stopping_rounds, verbose_eval=self.verbose)
                except:
                    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=self.clf.get_params()['n_estimators'], nfold=5,
                        metrics=[self.scoring], early_stopping_rounds=self.early_stopping_rounds)
                self.clf.set_params(n_estimators=cvresult.shape[0])
                self.clf.fit(current_df[self.predictors], current_df[self.target_column], eval_metric=self.scoring)

                #Predict training set:
                train_df_predictions = self.clf.predict(current_df[self.predictors])

                if self.target_type == 'binary':
                    train_df_predprob = self.clf.predict_proba(current_df[self.predictors])[:,1]
                    print("Accuracy : %.4g" % metrics.accuracy_score(current_df[self.target_column].values, train_df_predictions))
                    print("AUC Score (Train): %f" % metrics.roc_auc_score(current_df[self.target_column], train_df_predprob))
                elif self.target_type == 'linear':
                    print("Mean squared error: %f" % metrics.mean_squared_error(current_df[self.target_column].values, train_df_predictions))
                    print("Root mean squared error: %f" % np.sqrt(metrics.mean_squared_error(current_df[self.target_column].values, train_df_predictions)))
                filename = self.prefix + '_' + str(idx) + '.pkl'
                self.save(filename)

    def predict(self, test_df, return_multi_outputs=False, return_mean_std=False):
        print('### predicting ###')
        print('## preprocessing test set')
        if self.id_column in test_df:
            ids = test_df[self.id_column]
        if self.target_column in test_df.columns:
            targets = test_df[self.target_column]
        self.test_df = self.preprocess(test_df, train=False)
        if self.id_column in test_df:
            self.test_df[self.id_column] = ids
        if self.target_column in test_df.columns:
            self.test_df[self.target_column] = targets
        for col in self.predictors:
            if col not in self.test_df.columns:
                self.test_df[col] = np.nan

        # prediction
        print('## predicting from test set')
        output_list = []
        output = None
        for idx, ns in enumerate(range(self.n_samples)):
            if self.n_samples == 1:
                xgb = self
                if self.target_type == 'binary':
                    output = xgb.clf.predict_proba(self.test_df[self.predictors])[:,1]
                elif self.target_type == 'linear':
                    output = xgb.clf.predict(self.test_df[self.predictors])
            else:
                try:
                    filename = self.prefix + '_' + str(idx) + '.pkl'
                    xgb = self.load(filename)
                    if self.target_type == 'binary':
                        output = xgb.clf.predict_proba(self.test_df[self.predictors])[:,1]
                    elif self.target_type == 'linear':
                        output = xgb.clf.predict(self.test_df[self.predictors])
                    output_list.append(list(output))
                except IOError:
                    print('no file found, skipping')
        # average the outputs if n_samples is more than one
        if self.n_samples == 1:
            self.output = output
            try:
                self.multi_outputs = [list(output)]
            except:
                self.multi_outputs = None
        else:
            self.output = np.mean(output_list, axis=0)
            self.multi_outputs = output_list
        if return_multi_outputs:
            return self.multi_outputs
        elif return_mean_std:
            return (self.output, np.std(output_list, axis=0))
        else:
            return self.output

    def feature_importance(self, num_print=10, display=True):
        feature_importance = sorted(list(self.clf.booster().get_fscore().items()), key = operator.itemgetter(1), reverse=True)

        impt = pd.DataFrame(feature_importance)
        impt.columns = ['feature', 'importance']
        print(impt[:num_print])
        if display:
            impt[:num_print].plot("feature", "importance", kind="barh", color=sns.color_palette("deep", 3))


    def preprocess(self, df, train=True):
        # one hot encoding of categorical variables
        print('## one hot encoding of categorical variables')
        for col in self.categorical_columns:
            if self.verbose:
                print('one hot encoding: ', col)
            df = pd.concat([df, pd.get_dummies(df[col]).rename(columns=lambda x: col+'_'+str(x))], axis=1)
            df = df.drop([col], axis=1)

        # if training, determine columns to be removed
        if train:
            # drop columns that are too sparse to be informative
            self.cols_to_remove = []
            print('## dropping columns below sparsity threshold')
            for col in df.columns:
                nan_cnt = 0
                for x in df[col]:
                    try:
                        if np.isnan(x):
                            nan_cnt += 1
                    except:
                        pass
                if nan_cnt/float(len(df[col])) > 0.6: # arbitrary cutoff, if more than 60% missing then drop
                    if self.verbose:
                        print('will drop', col)
                    self.cols_to_remove.append(col)

            # drop columns that have no standard deviation (not informative)
            print('## dropping columns with no variation')
            for col in df.columns:
                if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                    if df[col].std() == 0:
                        print('will drop', col)
                        self.cols_to_remove.append(col)
        if self.verbose and self.cols_to_remove:
            print('dropping the following columns:', self.cols_to_remove)
            df = df.drop(self.cols_to_remove, axis=1)

        if self.verbose:
            print('## DataFrame shape is now:', df.shape)

        # convert to numerical where possible
        #print('## converting numerical data to numeric dtype')
        #df = df.convert_objects(convert_numeric=True)

        # convert columns specified to be int and float
        for col in self.numeric_columns:
            if self.verbose:
                print('converting', col)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if self.verbose:
                print(df[col].dtype)
        df = df.drop(self.drop_columns, axis=1)

        # drop all those that are object type
        print('## dropping non-numerical columns')
        for col in df.columns:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64' or df[col].dtype == 'bool':
                pass
            else:
                if self.verbose:
                    print('dropping because not int, float, or bool:', col)
                df = df.drop([col], axis=1)
        return df

    def _to_int(self, num):
        try:
            return int(num)
        except:
            return

    def _to_float(self, num):
        try:
            return float(num)
        except:
            return

    def random_sample(self, df, fraction=0.2, n_samples=None):
        """
        splits into random samples
        - n_samples: how many samples you want returned (default = All)
        - fraction : what fraction of data to include in the sample (default = 0.2)
        """
        print('constructing random samples')
        num_rows = len(df)
        len_sample = round(fraction * num_rows)
        # create list of slice index lists
        indices = range(0,num_rows)
        slice_list = []
        tmp_idx_list = []
        while len(indices) > 0:
            while len(tmp_idx_list) < len_sample and len(indices) > 0:
                idx = indices.pop(random.randrange(len(indices)))
                tmp_idx_list.append(idx)
            slice_list.append(tmp_idx_list)
            tmp_idx_list = []
        # get slices
        sample_list = []
        for s in range(n_samples):
            try:
                sample_list.append(df.loc[slice_list[s],:])
            except:
                pass
        return sample_list

    def write_csv(self, filename, include_actual=False):
        """
        write results to csv
        - include actual: if actual values are known for test set, and we want to print them
        """
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            headers = [self.id_column, self.target_column]
            if include_actual:
                headers.append('actual')
            writer.writerow(headers)
            try:
                for idx, value in enumerate(self.output):
                    test_id = self.test_df[self.id_column][idx]
                    test_output = self.output[idx]
                    to_write = [test_id, test_output]
                    if include_actual:
                        to_write.append(self.test_df[self.target_column][idx])
                    writer.writerow(to_write)
                print('results written to ' + filename)
            except:
                print('write_csv failed')

    def save(self, filename='xgb.pkl'):
        joblib.dump(self, filename)

    def load(self, model_file='xgb.pkl'):
        xgb = joblib.load(model_file)
        return xgb

