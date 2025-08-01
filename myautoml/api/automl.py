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
    def __init__(self, task=None, try_multilayer_stacking=True, stacking_layers=None):
        self.results = None
        self.models = {}
        self.ensemble_weights = {}
        self.logger = setup_logger('AutoML', os.path.join(config.MODEL_PATH, 'automl.log'))
        self.feature_engineer = FeatureEngineer()
        self.task = task  # 'classification' or 'regression', can be set or auto-detected in fit
        self.best_model = None
        self.ensemble_models = []
        self.selected_features = None
        self.try_multilayer_stacking = try_multilayer_stacking
        self.stacking_layers = stacking_layers

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        import traceback
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from myautoml.core.hpo_optuna import run_optuna_hpo
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

        # Define model configs and param_spaces for HPO
        model_configs = []
        if self.task == 'classification':
            from myautoml.models.xgboost_model import XGBoostClassifierModel
            from myautoml.models.lgbm_model import LGBMClassifierModel
            from myautoml.models.catboost_model import CatBoostClassifierModel
            from myautoml.models.neuralnet import NeuralNetClassifier
            n_classes = len(np.unique(y_enc))
            model_configs = [
                {
                    'name': 'xgboost',
                    'model_class': XGBoostClassifierModel,
                    'param_space': {
                        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 0, 5),
                        'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 0, 5),
                    },
                    'extra': {'n_classes': n_classes, 'input_dim': X_processed.shape[1]}
                },
                {
                    'name': 'lgbm',
                    'model_class': LGBMClassifierModel,
                    'param_space': {
                        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 0, 5),
                        'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 0, 5),
                    },
                    'extra': {'n_classes': n_classes, 'input_dim': X_processed.shape[1]}
                },
                {
                    'name': 'catboost',
                    'model_class': CatBoostClassifierModel,
                    'param_space': {
                        'iterations': lambda trial: trial.suggest_int('iterations', 100, 500),
                        'depth': lambda trial: trial.suggest_int('depth', 3, 10),
                        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'l2_leaf_reg': lambda trial: trial.suggest_float('l2_leaf_reg', 1, 10),
                        'bagging_temperature': lambda trial: trial.suggest_float('bagging_temperature', 0, 1),
                    },
                    'extra': {'n_classes': n_classes, 'input_dim': X_processed.shape[1]}
                },
                {
                    'name': 'neuralnet',
                    'model_class': NeuralNetClassifier,
                    'param_space': {
                        'input_dim': lambda trial: X_processed.shape[1],
                        'output_dim': lambda trial: n_classes,
                        'hidden_layers': lambda trial: [trial.suggest_int('hl1', 32, 128), trial.suggest_int('hl2', 16, 64)],
                        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                        'batch_size': lambda trial: trial.suggest_categorical('batch_size', [32, 64, 128]),
                        'epochs': lambda trial: trial.suggest_int('epochs', 20, 100),
                    },
                    'extra': {'n_classes': n_classes, 'input_dim': X_processed.shape[1]}
                },
            ]
            metric = 'accuracy'
        else:
            from myautoml.models.xgboost_model import XGBoostModel
            from myautoml.models.lgbm_model import LGBMModel
            from myautoml.models.catboost_model import CatBoostModel
            from myautoml.models.neuralnet import NeuralNetRegressor
            model_configs = [
                {
                    'name': 'xgboost',
                    'model_class': XGBoostModel,
                    'param_space': {
                        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 0, 5),
                        'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 0, 5),
                    },
                    'extra': {'input_dim': X_processed.shape[1]}
                },
                {
                    'name': 'lgbm',
                    'model_class': LGBMModel,
                    'param_space': {
                        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 0, 5),
                        'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 0, 5),
                    },
                    'extra': {'input_dim': X_processed.shape[1]}
                },
                {
                    'name': 'catboost',
                    'model_class': CatBoostModel,
                    'param_space': {
                        'iterations': lambda trial: trial.suggest_int('iterations', 100, 500),
                        'depth': lambda trial: trial.suggest_int('depth', 3, 10),
                        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'l2_leaf_reg': lambda trial: trial.suggest_float('l2_leaf_reg', 1, 10),
                        'bagging_temperature': lambda trial: trial.suggest_float('bagging_temperature', 0, 1),
                    },
                    'extra': {'input_dim': X_processed.shape[1]}
                },
                {
                    'name': 'neuralnet',
                    'model_class': NeuralNetRegressor,
                    'param_space': {
                        'input_dim': lambda trial: X_processed.shape[1],
                        'output_dim': lambda trial: 1,
                        'hidden_layers': lambda trial: [trial.suggest_int('hl1', 32, 128), trial.suggest_int('hl2', 16, 64)],
                        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                        'batch_size': lambda trial: trial.suggest_categorical('batch_size', [32, 64, 128]),
                        'epochs': lambda trial: trial.suggest_int('epochs', 20, 100),
                    },
                    'extra': {'input_dim': X_processed.shape[1]}
                },
            ]
            metric = 'neg_root_mean_squared_error'

        # HPO for each model (Bayesian + Hyperband), parallelized
        def hpo_for_model(config):
            try:
                # Bayesian Optimization (TPE)
                best_params, best_score, _ = run_optuna_hpo(
                    config['model_class'],
                    config['param_space'],
                    X_processed,
                    y_enc if self.task == 'classification' else y,
                    metric=metric,
                    direction='maximize' if self.task == 'classification' else 'maximize',
                    n_trials=30,
                    use_hyperband=False,
                    is_classification=(self.task == 'classification')
                )
                # Hyperband
                best_params_hb, best_score_hb, _ = run_optuna_hpo(
                    config['model_class'],
                    config['param_space'],
                    X_processed,
                    y_enc if self.task == 'classification' else y,
                    metric=metric,
                    direction='maximize' if self.task == 'classification' else 'maximize',
                    n_trials=30,
                    use_hyperband=True,
                    is_classification=(self.task == 'classification')
                )
                # Pick best of TPE/Hyperband
                if (self.task == 'classification' and best_score_hb > best_score) or (self.task == 'regression' and best_score_hb > best_score):
                    best_params = best_params_hb
                    best_score = best_score_hb
                # Instantiate model with best params
                model = config['model_class'](**{**best_params, **config.get('extra', {})})
                return (model, best_score, None)
            except Exception as e:
                return (None, None, traceback.format_exc())

        with ProcessPoolExecutor() as executor:
            future_to_config = {executor.submit(hpo_for_model, config): config for config in model_configs}
            best_score = -np.inf if self.task == 'classification' else float('inf')
            best_model = None
            best_type = None
            models = []
            for future in as_completed(future_to_config):
                model, score, err = future.result()
                if err or model is None:
                    print(f"Model {future_to_config[future]['name']} failed: {err}")
                    continue
                models.append(model)
                if self.task == 'classification':
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_type = 'single'
                else:
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_type = 'single'

        # Multi-layer stacking (if enabled)
        if self.try_multilayer_stacking and len(models) > 1:
            from myautoml.core.multilayer_stacking import MultiLayerStacking
            if self.stacking_layers is None:
                stacking_layers = [models, models]
            else:
                stacking_layers = self.stacking_layers
            stacking_model = MultiLayerStacking(stacking_layers, task=self.task)
            from myautoml.core.cross_validation import cross_val_score_model
            if self.task == 'classification':
                stacking_score = cross_val_score_model(
                    stacking_model, X_processed, y_enc, metric, n_splits=5, is_classification=True
                )
            else:
                stacking_score = cross_val_score_model(
                    stacking_model, X_processed, y, metric, n_splits=5, is_classification=False
                )
            if (
                (self.task == 'classification' and stacking_score > best_score)
                or (self.task == 'regression' and stacking_score > best_score)
            ):
                best_score = stacking_score
                best_model = stacking_model
                best_type = 'stacking'

        if best_model is None:
            raise RuntimeError("No model could be successfully trained. Check model wrappers and data.")
        # Fit the best model on the full data
        if best_type == 'stacking':
            if self.task == 'classification':
                best_model.fit(X_processed, y_enc)
            else:
                best_model.fit(X_processed, y)
        else:
            if self.task == 'classification':
                best_model.fit(X_processed, y_enc)
            else:
                best_model.fit(X_processed, y)
        self.best_model = best_model
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
            # Decode integer predictions back to string labels if label_encoder exists and preds are integer-encoded
            if (
                hasattr(self, 'label_encoder') and self.label_encoder is not None
                and (np.issubdtype(preds.dtype, np.integer) or np.issubdtype(preds.dtype, np.floating))
            ):
                preds = self.label_encoder.inverse_transform(preds.astype(int))
        return np.asarray(preds).ravel()

