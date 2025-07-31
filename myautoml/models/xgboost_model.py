from xgboost import XGBClassifier

class XGBoostClassifierModel:
    def __init__(self, params=None):
        self.params = params if params is not None else {}
        self.model = XGBClassifier(**self.params)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class XGBoostModel:
    def __init__(self, params=None):
        import xgboost as xgb
        self.params = params if params is not None else {}
        # Ensure regression objective is set
        self.params['objective'] = 'reg:squarederror'
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_model(self):
        return self.model

    def fit(self, X, y):
        return self.train(X, y)