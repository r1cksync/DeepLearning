import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flaml import AutoML as FLAMLAutoML
import joblib
import os
import sys

# Import your custom AutoML library
sys.path.append(os.path.abspath('myautoml'))
from api.automl import AutoML as MyAutoML

# Load dataset
DATA_PATH = '../Housing.csv' if not os.path.exists('Housing.csv') else 'Housing.csv'
df = pd.read_csv(DATA_PATH)

# Assume the last column is the target

# Encode categorical features

from pandas.api.types import is_object_dtype
X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]
cat_cols = [col for col in X_raw.columns if is_object_dtype(X_raw[col])]
if cat_cols:
    X = pd.get_dummies(X_raw, columns=cat_cols)
else:
    X = X_raw

# Encode target if categorical
if is_object_dtype(y_raw):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
else:
    y = y_raw

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

import numpy as np

# 1. Test your custom AutoML
my_automl = MyAutoML()
my_automl.fit(X_train, y_train)
y_pred_my = my_automl.predict(X_test)
results['MyAutoML'] = np.sqrt(mean_squared_error(y_test, y_pred_my))

# 2. Test FLAML
flaml_automl = FLAMLAutoML(task='regression', time_budget=60)
flaml_automl.fit(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test)
y_pred_flaml = flaml_automl.predict(X_test)
results['FLAML'] = np.sqrt(mean_squared_error(y_test, y_pred_flaml))

print('RMSE Results:')
for k, v in results.items():
    print(f'{k}: {v}')
