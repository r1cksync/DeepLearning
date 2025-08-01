import pandas as pd
import optuna
from catboost import CatBoostClassifier
from myautoml.core.hpo_optuna import run_optuna_hpo

df = pd.read_csv('train_new.csv')
TARGET_COL = df.columns[-1]
X = df.drop(['id', TARGET_COL], axis=1)
y = df[TARGET_COL]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)

param_space = {
    'iterations': lambda trial: trial.suggest_int('iterations', 100, 500),
    'depth': lambda trial: trial.suggest_int('depth', 3, 10),
    'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'l2_leaf_reg': lambda trial: trial.suggest_float('l2_leaf_reg', 1, 10),
    'bagging_temperature': lambda trial: trial.suggest_float('bagging_temperature', 0, 1),
}

best_params, best_score, study = run_optuna_hpo(
    CatBoostClassifier,
    param_space,
    X,
    y_enc,
    metric='accuracy',
    direction='maximize',
    n_trials=30,
    use_hyperband=False,
    is_classification=True
)
print('Optuna TPE best params:', best_params)
print('Optuna TPE best score:', best_score)

best_params_hb, best_score_hb, study_hb = run_optuna_hpo(
    CatBoostClassifier,
    param_space,
    X,
    y_enc,
    metric='accuracy',
    direction='maximize',
    n_trials=30,
    use_hyperband=True,
    is_classification=True
)
print('Optuna Hyperband best params:', best_params_hb)
print('Optuna Hyperband best score:', best_score_hb)
