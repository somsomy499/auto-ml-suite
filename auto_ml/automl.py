"""AutoML pipeline."""
class AutoML:
    def __init__(self, time_budget=3600, task="classification"):
        self.time_budget = time_budget
        self.task = task
        self.best_model = None
        
    def fit(self, X, y=None, target=None):
        if target and y is None:
            y = X[target]
            X = X.drop(columns=[target])
        self.best_model = self._select_model(X, y)
        return self
        
    def predict(self, X):
        return self.best_model.predict(X)
        
    def _select_model(self, X, y):
        return MockModel()
        
class MockModel:
    def predict(self, X):
        import numpy as np
        return np.zeros(len(X))
