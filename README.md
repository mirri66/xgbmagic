# xgbmagic
xgbmagic takes a automatically runs a standard workflow for using XGBoost to train a model based on a pandas dataframe.
- performs one-hot encoding for categorical features, 
- drops uninformative features (no variability, too many missing values...)
- trains the model
- plots most important features

## Installation
Install xgboost first
```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
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
