"""Automatic feature engineering transforms."""
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    def __init__(self):
        self.transforms = []
        
    def add_numeric(self, columns: List[str]):
        self.transforms.append(("numeric", columns))
        return self
        
    def add_interaction(self, col_a: str, col_b: str):
        self.transforms.append(("interaction", (col_a, col_b)))
        return self
        
    def add_polynomial(self, column: str, degree: int = 2):
        self.transforms.append(("polynomial", (column, degree)))
        return self
        
    def fit_transform(self, data: Dict[str, np.ndarray]):
        result = {}
        for name, values in data.items():
            result[name] = values
        for ttype, params in self.transforms:
            if ttype == "interaction":
                a, b = params
                result[f"{a}_x_{b}"] = data[a] * data[b]
            elif ttype == "polynomial":
                col, deg = params
                for d in range(2, deg + 1):
                    result[f"{col}_pow{d}"] = data[col] ** d
        return result
