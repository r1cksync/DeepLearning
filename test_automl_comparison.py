
import pandas as pd
import numpy as np
from flaml import AutoML as FLAMLAutoML
import os
import sys
sys.path.append(os.path.abspath('myautoml'))
from myautoml.api.automl import AutoML as MyAutoML

# Load new train and test data
train_df = pd.read_csv('train_new.csv')
test_df = pd.read_csv('test_new.csv')
sample_submission = pd.read_csv('sample_submission_new.csv')

# Prepare features and target
target_col = 'Fertilizer Name'
X_train = train_df.drop(['id', target_col], axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']

# --- MyAutoML (Classification) ---
my_automl = MyAutoML(task='classification')
my_automl.fit(X_train, y_train)
my_preds = my_automl.predict(X_test)
my_submission = pd.DataFrame({'id': test_ids, target_col: my_preds})
my_submission.to_csv('myautoml_submission_new.csv', index=False)
print('MyAutoML predictions saved to myautoml_submission_new.csv')

# --- FLAML (Classification) ---
flaml_automl = FLAMLAutoML(task='classification', time_budget=60)
flaml_automl.fit(X_train=X_train, y_train=y_train)
flaml_preds = flaml_automl.predict(X_test)
flaml_submission = pd.DataFrame({'id': test_ids, target_col: flaml_preds})
flaml_submission.to_csv('flaml_submission_new.csv', index=False)
print('FLAML predictions saved to flaml_submission_new.csv')
