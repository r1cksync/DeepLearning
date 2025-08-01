import pandas as pd
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor
from myautoml.core.hpo_optuna import run_optuna_hpo
from sklearn.preprocessing import LabelEncoder

def run_hpo(task_type='classification', data_path='train_new.csv'):
    df = pd.read_csv(data_path)
    TARGET_COL = df.columns[-1]
    X = df.drop(['id', TARGET_COL], axis=1)
    y = df[TARGET_COL]
    if task_type == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)
        model_cls = CatBoostClassifier
        metric = 'accuracy'
        direction = 'maximize'
        is_classification = True
    else:
        model_cls = CatBoostRegressor
        metric = 'neg_root_mean_squared_error'
        direction = 'maximize'
        is_classification = False
    param_space = {
        'iterations': lambda trial: trial.suggest_int('iterations', 100, 500),
        'depth': lambda trial: trial.suggest_int('depth', 3, 10),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': lambda trial: trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': lambda trial: trial.suggest_float('bagging_temperature', 0, 1),
    }
    # Bayesian Optimization (TPE)
    best_params, best_score, study = run_optuna_hpo(
        model_cls,
        param_space,
        X,
        y,
        metric=metric,
        direction=direction,
        n_trials=30,
        use_hyperband=False,
        is_classification=is_classification
    )
    print('Optuna TPE best params:', best_params)
    print('Optuna TPE best score:', best_score)
    # Hyperband
    best_params_hb, best_score_hb, study_hb = run_optuna_hpo(
        model_cls,
        param_space,
        X,
        y,
        metric=metric,
        direction=direction,
        n_trials=30,
        use_hyperband=True,
        is_classification=is_classification
    )
    print('Optuna Hyperband best params:', best_params_hb)
    print('Optuna Hyperband best score:', best_score_hb)

if __name__ == '__main__':
    # For classification
    run_hpo('classification', 'train_new.csv')
    # For regression
    run_hpo('regression', 'train.csv')
