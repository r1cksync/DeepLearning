# Stacking utility for classification and regression
from sklearn.ensemble import StackingClassifier, StackingRegressor

def stack_models(models, X, y, task='classification', meta_model=None, n_folds=5):
    """
    Fit a stacking ensemble using the provided base models and an optional meta-model.
    If meta_model is None, use LogisticRegression for classification, Ridge for regression.
    Returns the fitted stacking ensemble.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    estimators = [(f"model_{i}", m) for i, m in enumerate(models)]
    if task == 'classification':
        if meta_model is None:
            meta_model = LogisticRegression(max_iter=1000)
        stacker = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=n_folds, passthrough=False, n_jobs=-1)
    else:
        if meta_model is None:
            meta_model = Ridge()
        stacker = StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=n_folds, passthrough=False, n_jobs=-1)
    stacker.fit(X, y)
    return stacker
