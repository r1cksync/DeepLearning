import pandas as pd
import optuna
from lightgbm import LGBMRegressor
from myautoml.core.hpo_optuna import run_optuna_hpo

df = pd.read_csv('train.csv')
TARGET_COL = df.columns[-1]
X = df.drop(['id', TARGET_COL], axis=1)
y = df[TARGET_COL]

param_space = {
    'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
    'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
    'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 0, 5),
    'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 0, 5),
}

best_params, best_score, study = run_optuna_hpo(
    LGBMRegressor,
    param_space,
    X,
    y,
    metric='neg_root_mean_squared_error',
    direction='maximize',
    n_trials=30,
    use_hyperband=False,
    is_classification=False
)
print('Optuna TPE best params:', best_params)
print('Optuna TPE best score:', best_score)

best_params_hb, best_score_hb, study_hb = run_optuna_hpo(
    LGBMRegressor,
    param_space,
    X,
    y,
    metric='neg_root_mean_squared_error',
    direction='maximize',
    n_trials=30,
    use_hyperband=True,
    is_classification=False
)
print('Optuna Hyperband best params:', best_params_hb)
print('Optuna Hyperband best score:', best_score_hb)
