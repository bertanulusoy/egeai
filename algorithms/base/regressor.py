import numpy as np
import pandas as Series
import pandas as DataFrame


class Regressor:
    def __init__(self, X_train: DataFrame, X_valid: DataFrame, y_train: Series, y_valid: Series) -> None:
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.random_state = 42

    def best_number_of_estimator(self) -> int:
        pass

    def score(self):
        pass
