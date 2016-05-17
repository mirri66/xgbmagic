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

class Xgb:
    def __init__(self, df, target_column='', id_column='', target_type='binary', categorical_columns=[], drop_columns=[], numeric_columns=[], num_training_rounds=500, verbose=1, early_stopping_rounds=None):
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
                self.num_training_rounds = num_training_rounds
                # init the classifier
                if self.target_type == 'binary':
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

        xgtrain  = xgb.DMatrix(self.df[self.predictors], label=self.df[self.target_column], missing=np.nan)
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
        self.clf.fit(self.df[self.predictors], self.df[self.target_column],eval_metric=self.scoring)

        #Predict training set:
        train_df_predictions = self.clf.predict(self.df[self.predictors])

        if self.target_type == 'binary':
            train_df_predprob = self.clf.predict_proba(self.df[self.predictors])[:,1]
            print("Accuracy : %.4g" % metrics.accuracy_score(self.df[self.target_column].values, train_df_predictions))
            print("AUC Score (Train): %f" % metrics.roc_auc_score(self.df[self.target_column], train_df_predprob))
        elif self.target_type == 'linear':
            print("Mean squared error: %f" % metrics.mean_squared_error(self.df[self.target_column].values, train_df_predictions))
            print("Root mean squared error: %f" % np.sqrt(metrics.mean_squared_error(self.df[self.target_column].values, train_df_predictions)))

    def predict(self, test_df):
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

        if self.target_type == 'binary':
            self.output = self.clf.predict_proba(self.test_df[self.predictors])[:,1]
        elif self.target_type == 'linear':
            self.output = self.clf.predict(self.test_df[self.predictors])
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
            if col not in self.cols_to_remove:
                if self.verbose:
                    print('converting', col)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if self.verbose:
                    print(df[col].dtype)

        # drop those marked for dropping
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
            for idx, value in enumerate(self.output):
                test_id = self.test_df[self.id_column][idx]
                test_output = self.output[idx]
                to_write = [test_id, test_output]
                if include_actual:
                    to_write.append(self.test_df[self.target_column][idx])
                writer.writerow(to_write)

    def save(self, filename='xgb.pkl'):
        joblib.dump(self, filename)
