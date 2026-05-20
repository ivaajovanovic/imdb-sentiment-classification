import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, C: float = 1.0, max_iter: int = 1000, solver: str = 'lbfgs') -> None:
        self._model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)

    @property
    def sklearn_model(self) -> LogisticRegression:
        return self._model

    @sklearn_model.setter
    def sklearn_model(self, model: LogisticRegression) -> None:
        self._model = model

    def train(self, X_train: sp.spmatrix | np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X_test: sp.spmatrix | np.ndarray) -> np.ndarray:
        return self._model.predict(X_test)