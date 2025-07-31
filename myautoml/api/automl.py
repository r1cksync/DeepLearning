import os
import numpy as np
import pandas as pd
from myautoml.core.feature_engineering import FeatureEngineer, handle_missing_values, encode_categorical_features, scale_features
from myautoml.core.model_selector import ModelSelector
from myautoml.core.hyperparameter_optimization import HyperparameterOptimizer
from myautoml.core.ensembling import blend_models
from myautoml.core.evaluator import Evaluator
from myautoml.utils.metrics import accuracy_score, f1_score
from myautoml.utils.logger import setup_logger, log_message
from myautoml.utils import config
from myautoml.models.xgboost_model import XGBoostModel
from myautoml.models.lgbm_model import LGBMModel
from myautoml.models.catboost_model import CatBoostModel

class AutoML:
    def __init__(self):
        self.results = None
        self.models = {}
        self.ensemble_weights = {}
        self.logger = setup_logger('AutoML', os.path.join(config.MODEL_PATH, 'automl.log'))

        self.feature_engineer = FeatureEngineer()
        self.models = [
            XGBoostModel(params=config.XGBOOST_PARAMS),
            LGBMModel(params=config.LGBM_PARAMS),
            CatBoostModel(params=config.CATBOOST_PARAMS)
        ]
        self.best_model = None
        self.ensemble_models = []
        self.selected_features = None
        # Define hyperparameter grids for each model
        param_grids = {
            'XGBoostModel': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'objective': ['reg:squarederror']
            },
            'LGBMModel': {
                'n_estimators': [50, 100, 200],
                'max_depth': [-1, 5, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'CatBoostModel': {
                'iterations': [100, 200, 500],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }

        model_wrappers = [
            (XGBoostModel(params=config.XGBOOST_PARAMS), param_grids['XGBoostModel']),
            (LGBMModel(params=config.LGBM_PARAMS), param_grids['LGBMModel']),
            (CatBoostModel(params=config.CATBOOST_PARAMS), param_grids['CatBoostModel'])
        ]

    def fit(self, X, y):
        # Fit feature engineering pipeline on train, transform train
        X_processed = self.feature_engineer.fit_transform(X, y)
        self.selected_features = X_processed.columns.tolist()

        # Model selection (simple RMSE-based selection)
        best_score = float('inf')
        for model in self.models:
            model.fit(X_processed, y)
            preds = model.predict(X_processed)
            score = Evaluator.rmse(y, preds)
            if score < best_score:
                best_score = score
                self.best_model = model

        # Optionally, fit ensemble on train
        self.ensemble_models = [self.best_model]

    def predict(self, X):
        # Transform test using fitted feature engineering pipeline
        X_processed = self.feature_engineer.transform(X)
        # Add missing columns (from train) as zeros, drop extra columns
        for col in self.selected_features:
            if col not in X_processed:
                X_processed[col] = 0
        X_processed = X_processed[self.selected_features]
        # Ensure all columns are numeric
        X_processed = X_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
        preds = self.best_model.predict(X_processed)
        return preds