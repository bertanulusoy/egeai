import numpy as np
import pandas as pd


class Regressor:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.X_train = X_train
