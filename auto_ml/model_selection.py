"""Automated model selection."""
from dataclasses import dataclass
from typing import List

@dataclass
class ModelScore:
    name: str
    score: float
    train_time: float
    params: dict

class ModelSelector:
    def __init__(self, task: str = "classification"):
        self.task = task
        self.candidates = []
        
    def add_candidate(self, name, model_class, params=None):
        self.candidates.append({"name": name, "class": model_class, "params": params or {}})
        
    def select(self, X_train, y_train, X_val, y_val) -> ModelScore:
        best = None
        for cand in self.candidates:
            import time
            start = time.time()
            model = cand["class"](**cand["params"])
            model.fit(X_train, y_train)
            train_time = time.time() - start
            score = model.score(X_val, y_val)
            ms = ModelScore(cand["name"], score, train_time, cand["params"])
            if best is None or ms.score > best.score:
                best = ms
        return best
