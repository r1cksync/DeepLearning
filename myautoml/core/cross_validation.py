
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def cross_val_score_model(model, X, y, metric, n_splits=5, is_classification=True, fit_kwargs=None):
    """
    Perform K-Fold cross-validation and return the mean score.
    """
    if is_classification:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    fit_kwargs = fit_kwargs or {}
    for train_idx, val_idx in kf.split(X, y if is_classification else None):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model_clone = clone_model(model)
        model_clone.fit(X_train, y_train, **fit_kwargs)
        preds = model_clone.predict(X_val)
        # If metric expects probabilities, handle here if needed
        score = metric(y_val, preds)
        scores.append(score)
    return np.mean(scores)

def clone_model(model):
    # Try to use sklearn's clone if possible, else fallback to model's own copy method
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        import copy
        return copy.deepcopy(model)
