from abc import ABC, abstractmethod


class BaseAutoencoder(ABC):

    @abstractmethod
    def train(self, X_train, X_val) -> dict:
        pass

    @abstractmethod
    def encode(self, X):
        pass

    @abstractmethod
    def reconstruct(self, X):
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass