# Stacking utility for classification and regression
from sklearn.ensemble import StackingClassifier, StackingRegressor

def blend_models(models, X, task='classification'):
    """
    Blend predictions from multiple models by averaging (regression) or majority vote (classification).
    Returns the blended predictions.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    def predict_model(model):
        return model.predict(X)
    with ThreadPoolExecutor() as executor:
        predictions = list(executor.map(predict_model, models))
    predictions = np.array(predictions)
    if task == 'classification':
        # Majority vote
        from scipy.stats import mode
        blended = mode(predictions, axis=0, keepdims=False)[0]
        return blended
    else:
        # Average
        return np.mean(predictions, axis=0)