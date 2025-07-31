class LGBMModel:
    def __init__(self, params=None):
        import lightgbm as lgb
        self.params = params if params is not None else {}
        self.model = lgb.LGBMRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def fit(self, X, y):
        return self.train(X, y)