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
    def __init__(self, task=None):
        self.results = None
        self.models = {}
        self.ensemble_weights = {}
        self.logger = setup_logger('AutoML', os.path.join(config.MODEL_PATH, 'automl.log'))
        self.feature_engineer = FeatureEngineer()
        self.task = task  # 'classification' or 'regression', can be set or auto-detected in fit
        self.best_model = None
        self.ensemble_models = []
        self.selected_features = None

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        # Auto-detect task if not set
        if self.task is None:
            if y.dtype == 'object' or y.nunique() <= 20:
                self.task = 'classification'
            else:
                self.task = 'regression'

        self.label_encoder = None
        y_enc = y
        if self.task == 'classification':
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y)

        X_processed = self.feature_engineer.fit_transform(X, y)
        self.selected_features = X_processed.columns.tolist()

        # Choose models based on task
        models = []
        if self.task == 'classification':
            from myautoml.models.xgboost_model import XGBoostClassifierModel
            from myautoml.models.lgbm_model import LGBMClassifierModel
            from myautoml.models.catboost_model import CatBoostClassifierModel
            from myautoml.models.neuralnet import NeuralNetClassifier
            n_classes = len(np.unique(y_enc))
            models = [
                XGBoostClassifierModel(),
                LGBMClassifierModel(),
                CatBoostClassifierModel(),
                NeuralNetClassifier(input_dim=X_processed.shape[1], output_dim=n_classes, hidden_layers=[64, 32])
            ]
            metric = accuracy_score
        else:
            from myautoml.models.xgboost_model import XGBoostModel
            from myautoml.models.lgbm_model import LGBMModel
            from myautoml.models.catboost_model import CatBoostModel
            from myautoml.models.neuralnet import NeuralNetRegressor
            models = [
                XGBoostModel(),
                LGBMModel(),
                CatBoostModel(),
                NeuralNetRegressor(input_dim=X_processed.shape[1], output_dim=1, hidden_layers=[64, 32])
            ]
            metric = Evaluator.rmse

        # Model selection
        best_score = -np.inf if self.task == 'classification' else float('inf')
        self.best_model = None
        for model in models:
            try:
                # Always fit and score with integer-encoded labels for classification
                if self.task == 'classification':
                    model.fit(X_processed, y_enc)
                    preds = model.predict(X_processed)
                    score = metric(y_enc, preds)
                    if score > best_score:
                        best_score = score
                        self.best_model = model
                else:
                    model.fit(X_processed, y)
                    preds = model.predict(X_processed)
                    score = metric(y, preds)
                    if score < best_score:
                        best_score = score
                        self.best_model = model
            except Exception as e:
                print(f"Model {type(model).__name__} failed: {e}")
        if self.best_model is None:
            raise RuntimeError("No model could be successfully trained. Check model wrappers and data.")
        self.ensemble_models = [self.best_model]

        # Model selection
        best_score = -np.inf if self.task == 'classification' else float('inf')
        self.best_model = None
        for model in models:
            try:
                model.fit(X_processed, y)
                preds = model.predict(X_processed)
                if self.task == 'classification':
                    score = metric(y, preds)
                    if score > best_score:
                        best_score = score
                        self.best_model = model
                else:
                    score = metric(y, preds)
                    if score < best_score:
                        best_score = score
                        self.best_model = model
            except Exception as e:
                print(f"Model {type(model).__name__} failed: {e}")
        if self.best_model is None:
            raise RuntimeError("No model could be successfully trained. Check model wrappers and data.")
        self.ensemble_models = [self.best_model]

    def predict(self, X):
        X_processed = self.feature_engineer.transform(X)
        for col in self.selected_features:
            if col not in X_processed:
                X_processed[col] = 0
        X_processed = X_processed[self.selected_features]
        X_processed = X_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
        preds = self.best_model.predict(X_processed)
        if self.task == 'classification':
            # If model outputs probabilities, convert to class labels
            if isinstance(preds, np.ndarray) and preds.ndim > 1 and preds.shape[1] > 1:
                preds = np.argmax(preds, axis=1)
            # Decode integer predictions back to string labels if label_encoder exists
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                preds = self.label_encoder.inverse_transform(preds.astype(int))
        return preds

