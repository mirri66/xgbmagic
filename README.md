# xgbmagic
*Pandas dataframe goes in, XGBoost model results come out*

The feature engineering step (creating new features and selectively removing unwanted features) is the most creative and fun step of training a model, whereas what follows is usually a standard data-processing flow.

Once you're done engineering your features, xgbmagic automatically runs a standard workflow for using XGBoost to train a model on a pandas dataframe.
- performs one-hot encoding for categorical features, 
- drops uninformative features (no variability, too many missing values...)
- trains the model
- plots the most important features in order of importance.

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
* categorical_columns (list of strings): a list of column names of columns containing categorical data
* verbose (boolean): verbosity of printouts. True = verbose


## Example
```
import xgbmagic
import pandas as pd

df = pd.read_csv('train.csv')

target_type = 'binary'

xgb = xgbmagic.Xgb(df, target_column='TARGET', id_column='ID', categorical_columns=[], num_training_rounds=1000, target_type=target_type, early_stopping_rounds=50)
xgb.train()

test_df = pd.read_csv('test.csv')
print(xgb.feature_importance())
output = xgb.predict(test_df)
xgb.write_csv('output-xgbmagic.csv')
```

## Issues
Please report issues and feedback [here](https://github.com/mirri66/xgbmagic/issues)

