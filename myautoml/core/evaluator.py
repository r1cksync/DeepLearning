class Evaluator:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def evaluate(self):
        predictions = self.model.predict(self.X)
        accuracy = self.accuracy_score(self.y, predictions)
        return accuracy

    def cross_validate(self, cv=5):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, self.X, self.y, cv=cv)
        return scores.mean()

    def accuracy_score(self, y_true, y_pred):
        return sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0

    @staticmethod
    def rmse(y_true, y_pred):
        import numpy as np
        return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))