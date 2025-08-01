import optuna
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import cross_val_score
import numpy as np

def run_optuna_hpo(model_class, param_space, X, y, metric, direction='maximize', n_trials=20, use_hyperband=False, is_classification=True):
    def objective(trial):
        params = {k: v(trial) for k, v in param_space.items()}
        model = model_class(**params)
        score = cross_val_score(model, X, y, scoring=metric, cv=3, n_jobs=-1)
        return np.mean(score)

    pruner = HyperbandPruner() if use_hyperband else None
    study = optuna.create_study(direction=direction, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value
    return best_params, best_score, study

# Example param_space:
# param_space = {
#     'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 200),
#     'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
# }
