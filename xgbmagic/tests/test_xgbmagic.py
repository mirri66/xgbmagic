import unittest
import pandas as pd
import xgbmagic
import random

n=100
data_dict = {'id':[str(a) for a in range(n)], 'y': [random.randrange(x+10) for x in range(n)], 'x1': [random.randrange(20+x*2) for x in range(n)], 'x2': [random.randrange(11) for x in range(n)]}
train_df = pd.DataFrame.from_dict(data_dict, orient='columns')
test_dict = {'id':['12345','12021'], 'y':[22,21], 'x1':[11,11], 'x2':[34, 123]}
test_df = pd.DataFrame.from_dict(test_dict, orient='columns')

bin_data_dict = {'id':[str(a) for a in range(n)], 'y': [random.randint(0,1) for x in range(n)], 'x1': [random.randrange(20+x*2) for x in range(n)], 'x2': [random.randrange(11) for x in range(n)]}
bin_train_df = pd.DataFrame.from_dict(bin_data_dict, orient='columns')
bin_test_dict = {'id':['12345','12021'], 'y':[0,1], 'x1':[11,11], 'x2':[34, 123]}
bin_test_df = pd.DataFrame.from_dict(bin_test_dict, orient='columns')


class TestXGB(unittest.TestCase):
    def test_init(self):
        drop_list = []
        numeric_list = []
        cat_list = []
        xgb = xgbmagic.Xgb(train_df, target_column='y', id_column='id', numeric_columns=numeric_list, drop_columns=drop_list, categorical_columns=cat_list, num_training_rounds=100, target_type='linear', verbose=True, prefix='test', sample_fraction=0.01, n_samples=2)
        self.assertIsInstance(xgb, xgbmagic.Xgb)

    def test_random_sample(self):
        n_samples = 3
        fraction = 0.2
        drop_list = []
        numeric_list = []
        cat_list = []
        xgb = xgbmagic.Xgb(train_df, target_column='y', id_column='id', numeric_columns=numeric_list, drop_columns=drop_list, categorical_columns=cat_list, num_training_rounds=100, target_type='linear', verbose=True, prefix='test', sample_fraction=0.01, n_samples=2)
        samples = xgb.random_sample(train_df, fraction=fraction, n_samples=n_samples)
        self.assertTrue(len(samples)==n_samples)
        self.assertTrue(len(samples[0])==round(fraction*len(train_df)))

    def test_output(self):
        drop_list = []
        numeric_list = []
        cat_list = []
        xgb = xgbmagic.Xgb(train_df, target_column='y', id_column='id', numeric_columns=numeric_list, drop_columns=drop_list, categorical_columns=cat_list, num_training_rounds=100, target_type='linear', verbose=True, prefix='test', sample_fraction=0.3, n_samples=2)
        xgb.train()
        output = xgb.predict(test_df)
        print 'OUTPUT', output
        self.assertTrue(len(output) == len(test_dict['id']))

    def test_binary(self):
        print(bin_train_df.head())
        drop_list = []
        numeric_list = []
        cat_list = []
        xgb = xgbmagic.Xgb(bin_train_df, target_column='y', id_column='id', numeric_columns=numeric_list, drop_columns=drop_list, categorical_columns=cat_list, num_training_rounds=100, target_type='binary', verbose=True, prefix='test', sample_fraction=0.3, n_samples=2)
        xgb.train()
        output = xgb.predict(bin_test_df)
        print 'OUTPUT', output
        self.assertTrue(len(output) == len(bin_test_dict['id']))


if __name__ == '__main__':
    unittest.main()
