class HyperparameterOptimizer:
    def __init__(self, model, param_grid, scoring='accuracy', n_iter=10, random_state=None):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None

    def optimize(self, X, y):
        from sklearn.model_selection import RandomizedSearchCV

        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            n_iter=self.n_iter,
            scoring=self.scoring,
            random_state=self.random_state,
            cv=5,
            verbose=1
        )
        search.fit(X, y)
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        return self.best_params_, self.best_score_

    def get_best_params(self):
        return self.best_params_

    def get_best_score(self):
        return self.best_score_