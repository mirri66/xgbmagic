# xgbmagic
*Pandas dataframe goes in, XGBoost model results come out*

The feature engineering step (creating new features and selectively removing unwanted features) is the most creative and fun step of training a model, whereas what follows is usually a standard data-processing flow.

Once you're done engineering your features, xgbmagic automatically runs a standard workflow for using XGBoost to train a model on a pandas dataframe.
- performs one-hot encoding for categorical features, 
- drops uninformative features (no variability, too many missing values...)
- trains the model
- plots the most important features in order of importance.

New features:
- allows for training on random sample(s) - e.g. if you'd like to train a subset of your data
- allows for training multiple times on different subsamples, then returning average/consensus from all models as the prediction

#### To do
- detect highly correlated columns and remove redundant columns
- remove categorical features with too many possible category values (to remove unhelpful features like names and ids)
- parameter tuning with GridsearchCV
- allow custom values for more parameters

## Installation
Install xgboost first
```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; sudo python setup.py install
```
Then install xgbmagic
```
pip install xgbmagic
```

## Documentation
Input parameters:
* df (DataFrame): pandas dataframe
* target_column (string): name of column containing the target parameter
* id_column (string): name of column containing IDs
* target_type (string): 'binary' for binary targets (classification), 'linear' for continuous targets (linear regression)
* categorical_columns (list of strings): a list of names of columns containing categorical data
* numerical_columns (list of strings): a list of names of columns containing numerical data
* drop_columns (list of strings): a list of names of columns to drop
* verbose (boolean): verbosity of printouts. True = verbose
* sample_fraction (float, 0-1): if this is not 1, a subsample of the data will be used to train the model
* n_samples (int): if this is more than 1, this number of models will be trained, each iteration trained on a subsample of data (size of sample determined by sample_fraction). Note that if n_samples * sample_fraction is more than one, the maximum number of models will be trained based on how many samples we can get from the stated sample_fraction (e.g. if sample_fraction = 0.2, the maximum by number of samples will be 5)
* prefix (string): prefix for names of the output files


## Example
```
import xgbmagic
import pandas as pd

# read the training data
df = pd.read_csv('train.csv')

# for logistic regression, target_type is 'binary'
target_type = 'binary'

# set columns that are categorical, numeric, and to be dropped here.
xgb = xgbmagic.Xgb(df, target_column='TARGET', id_column='ID', target_type=target_type, categorical_columns=[], drop_columns=[], numeric_columns=[], num_training_rounds=500, verbose=1, early_stopping_rounds=50)
xgb.train()

# use the model to predict values for the test set
test_df = pd.read_csv('test.csv')
print(xgb.feature_importance())
output = xgb.predict(test_df)

# write to csv
xgb.write_csv('output-xgbmagic.csv')

# save model
xgb.save('xgbmodel.pkl')

# load model
from sklearn.externals import joblib
xgb = joblib.load('xgbmodel.pkl')




```

## Issues
Please report issues and feedback [here](https://github.com/mirri66/xgbmagic/issues)

