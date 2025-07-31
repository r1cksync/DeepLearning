from catboost import CatBoostClassifier, CatBoostRegressor

class CatBoostClassifierModel:
    def __init__(self, params=None):
        self.params = params if params is not None else {}
        self.model = CatBoostClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

class CatBoostModel:
    def __init__(self, params=None):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(**(params if params else {}))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def fit(self, X, y):
        return self.train(X, y)