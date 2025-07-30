class XGBoostModel:
    def __init__(self, params=None):
        import xgboost as xgb
        self.params = params if params is not None else {}
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def get_model(self):
        return self.model