# Configuration settings for the AutoML project

# Paths
DATA_PATH = "data/"
MODEL_PATH = "models/"
REPORT_PATH = "reports/"

# Model parameters

XGBOOST_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 100,
    'objective': 'reg:squarederror'
}


LGBM_PARAMS = {
    'learning_rate': 0.1,
    'num_leaves': 31,
    'n_estimators': 100,
    'objective': 'regression'
}


CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'RMSE'
}

# Evaluation metrics
EVALUATION_METRICS = ['accuracy', 'f1_score', 'roc_auc']