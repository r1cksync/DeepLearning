import numpy as np
from sklearn.base import clone

class MultiLayerStacking:
    def __init__(self, layers, task='classification'):
        """
        layers: List of lists of models. Each sublist is a layer.
        task: 'classification' or 'regression'
        """
        self.layers = layers
        self.task = task
        self.fitted_layers = []

    def fit(self, X, y):
        X_current = X.copy()
        self.fitted_layers = []
        for layer_models in self.layers:
            layer_fitted = []
            layer_preds = []
            for model in layer_models:
                m = clone(model)
                m.fit(X_current, y)
                layer_fitted.append(m)
                preds = m.predict(X_current)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                layer_preds.append(preds)
            # Concatenate predictions from all models in this layer as new features
            X_current = np.hstack([X_current] + layer_preds)
            self.fitted_layers.append(layer_fitted)
        return self

    def predict(self, X):
        X_current = X.copy()
        for layer_fitted in self.fitted_layers:
            layer_preds = []
            for m in layer_fitted:
                preds = m.predict(X_current)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                layer_preds.append(preds)
            X_current = np.hstack([X_current] + layer_preds)
        # For final prediction, average (regression) or majority vote (classification) across last layer
        final_preds = np.column_stack([m.predict(X_current) for m in self.fitted_layers[-1]])
        if self.task == 'classification':
            from scipy.stats import mode
            return mode(final_preds, axis=1, keepdims=False)[0].ravel()
        else:
            return np.mean(final_preds, axis=1)
