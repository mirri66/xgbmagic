# xgbmagic
xgbmagic takes a pandas dataframe and uses XGBoost to train a model.

Performs one-hot encoding for categorical features, and drops uninformative features.

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
