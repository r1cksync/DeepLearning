class ModelSelector:
    def __init__(self, models, metrics):
        self.models = models
        self.metrics = metrics

    def select_model(self, X_train, y_train, X_val, y_val):
        best_model = None
        best_score = float('-inf')

        for model in self.models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = self.metrics.evaluate(predictions, y_val)

            if score > best_score:
                best_score = score
                best_model = model

        return best_model

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        return self.metrics.evaluate(predictions, y_test)