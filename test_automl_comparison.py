

import pandas as pd
import numpy as np
from flaml import AutoML as FLAMLAutoML
import os
import sys
sys.path.append(os.path.abspath('myautoml'))
from myautoml.api.automl import AutoML as MyAutoML

# Load regression train and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Prepare features and target for regression
target_col = 'price'
X_train = train_df.drop(['id', target_col], axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']

# --- MyAutoML (Regression) ---
my_automl = MyAutoML(task='regression')
my_automl.fit(X_train, y_train)
my_preds = my_automl.predict(X_test)
my_submission = pd.DataFrame({'id': test_ids, target_col: my_preds})
my_submission.to_csv('myautoml_submission.csv', index=False)
print('MyAutoML predictions saved to myautoml_submission.csv')

# --- FLAML (Regression) ---
flaml_automl = FLAMLAutoML(task='regression', time_budget=60)
flaml_automl.fit(X_train=X_train, y_train=y_train)
flaml_preds = flaml_automl.predict(X_test)
flaml_submission = pd.DataFrame({'id': test_ids, target_col: flaml_preds})
flaml_submission.to_csv('flaml_submission.csv', index=False)
print('FLAML predictions saved to flaml_submission.csv')
