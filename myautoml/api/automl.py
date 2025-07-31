
import os
import numpy as np
import pandas as pd
from myautoml.core.feature_engineering import handle_missing_values, encode_categorical_features, scale_features
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

    def fit(self, X, y):
        # 1. Feature Engineering
        log_message(self.logger, 'Starting feature engineering...')
        X = handle_missing_values(X)
        X = encode_categorical_features(X)
        X = scale_features(X)

        # 2. Model Wrappers
        log_message(self.logger, 'Initializing model wrappers...')
        model_wrappers = [
            XGBoostModel(params=config.XGBOOST_PARAMS),
            LGBMModel(params=config.LGBM_PARAMS),
            CatBoostModel(params=config.CATBOOST_PARAMS)
        ]

        # 3. Hyperparameter Optimization
        log_message(self.logger, 'Starting hyperparameter optimization...')
        optimized_models = []
        for wrapper in model_wrappers:
            optimizer = HyperparameterOptimizer(wrapper.model, wrapper.model.get_params(), scoring='neg_root_mean_squared_error', n_iter=5, random_state=42)
            best_params, _ = optimizer.optimize(X, y)
            wrapper.model.set_params(**best_params)
            optimized_models.append(wrapper)

        # 4. Model Selection
        log_message(self.logger, 'Selecting best model(s)...')
        selector = ModelSelector([w.model for w in optimized_models], metrics=Evaluator)
        # For simplicity, use train/val split here
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        best_model = selector.select_model(X_train, y_train, X_val, y_val)
        self.models = {'best': best_model}

        # 5. Ensembling (blending)
        log_message(self.logger, 'Blending models...')
        self.ensemble_models = [w.model for w in optimized_models]
        # Optionally, you can use stacking from ensembling.py as well

        # 6. Evaluation
        log_message(self.logger, 'Evaluating ensemble...')
        blended_pred = blend_models(self.ensemble_models, X_val, y_val)
        self.results = {
            'accuracy': accuracy_score(y_val, blended_pred),
            'f1_score': f1_score(y_val, blended_pred)
        }

    def predict(self, X):
        # Apply same feature engineering as in fit
        X = handle_missing_values(X)
        X = encode_categorical_features(X)
        X = scale_features(X)
        # Use blended ensemble for prediction
        return blend_models(self.ensemble_models, X, None)

    def run_automl(self, data):
        # Logic to run the AutoML process on the provided data
        pass

    def get_results(self):
        # Logic to return the results of the AutoML process
        return self.results

    def predict(self, X):
        import numpy as np
        if not self.models:
            raise Exception("No models trained. Call fit() first.")
        preds = []
        for name, model in self.models.items():
            preds.append(model.predict(X))
        preds = np.array(preds)
        # Weighted average
        weights = np.array([self.ensemble_weights[name] for name in self.models])
        return np.average(preds, axis=0, weights=weights)

    def run_automl(self, data):
        # Logic to run the AutoML process on the provided data
        pass

    def get_results(self):
        # Logic to return the results of the AutoML process
        return self.results