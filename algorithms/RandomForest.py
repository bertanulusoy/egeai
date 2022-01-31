from egeai.algorithms.base.regressor import Regressor

import pandas as Series
import pandas as DataFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class RandomForest(Regressor):
    def __init__(self, X_train: DataFrame, X_valid: DataFrame, y_train: Series, y_valid: Series):
        print("sfsfsdfsdfs")
        super().__init__(X_train, X_valid, y_train, y_valid)

    def __repr__(self):
        return 'RandomForest(*{!r})'.format(self.X_train)

    def best_number_of_estimator(self) -> int:
        return 10

    def score(self):
        n_estimators = self.best_number_of_estimator()
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_valid)
        return mean_absolute_error(self.y_valid, preds)
