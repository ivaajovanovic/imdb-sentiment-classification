import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from models.base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: int | None = None,
                 min_samples_split: int = 2, n_jobs: int = -1) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
        )

    @property
    def sklearn_model(self) -> RandomForestClassifier:
        return self._model

    @sklearn_model.setter
    def sklearn_model(self, model: RandomForestClassifier) -> None:
        self._model = model

    def train(self, X_train: sp.spmatrix | np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X_test: sp.spmatrix | np.ndarray) -> np.ndarray:
        return self._model.predict(X_test)