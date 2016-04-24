# xgbmagic
*Pandas dataframe goes in, XGBoost model results come out*

The feature engineering step (creating new features and selectively removing unwanted features) is the most creative and fun step of training a model, whereas what follows is usually a standard data-processing flow.

Once you're done engineering your features, xgbmagic automatically runs a standard workflow for using XGBoost to train a model on a pandas dataframe.
- performs one-hot encoding for categorical features, 
- drops uninformative features (no variability, too many missing values...)
- trains the model
- plots the most important features in order of importance.


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


## Examples
```
import xgbmagic
import pandas as pd

train_df = pd.read_csv('train.csv')
xgb = xgbmagic.Xgb(train_df)
xgb.train()
xgb.predict()
```

## Issues
Please report issues and feedback [here](https://github.com/mirri66/xgbmagic/issues)
