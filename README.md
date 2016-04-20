# xgbmagic

## Documentation


## Examples
```
import xgbmagic
import pandas as pd

train_df = pd.read_csv('train.csv')
xgb = xgbmagic.Xgb(train_df)
xgb.train()
xgb.predict()
```
