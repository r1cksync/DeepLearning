class CatBoostModel:
    def __init__(self, params=None):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(**(params if params else {}))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)