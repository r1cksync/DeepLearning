def accuracy_score(y_true, y_pred):
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct_predictions / len(y_true) if len(y_true) > 0 else 0.0

def f1_score(y_true, y_pred):
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0