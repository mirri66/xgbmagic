import xgbmagic
import pandas as pd

df = pd.read_csv('train.csv')

xgb = xgbmagic.Xgb(df, target_column='target', id_column='id', categorical_columns=['category_one', 'category_two'], num_training_rounds=100, target_type='binary')
xgb.train()
print(xgb.predict(df))
print(xgb.feature_importance())

