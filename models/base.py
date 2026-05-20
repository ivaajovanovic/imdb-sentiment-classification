from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseModel(ABC, BaseEstimator, ClassifierMixin):
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @property
    @abstractmethod
    def sklearn_model(self):
        pass

    def fit(self, X, y):
        """sklearn-compatible alias for train()."""
        self.train(X, y)
        return self

    def get_params(self, deep: bool = True) -> dict:
        """Return hyperparameters — required by GridSearchCV."""
        return {k: v for k, v in self._model.get_params(deep=deep).items()
                if k in self._init_params()}

    def set_params(self, **params):
        """Set hyperparameters — required by GridSearchCV."""
        self._model.set_params(**params)
        return self

    def _init_params(self) -> list:
        """Return list of param names exposed in __init__."""
        import inspect
        sig = inspect.signature(self.__class__.__init__)
        return [p for p in sig.parameters if p != 'self']