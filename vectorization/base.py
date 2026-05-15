from abc import ABC, abstractmethod

class BaseVectorizer(ABC):
    @abstractmethod
    def fit_transform(self, df):
        pass
    
    @abstractmethod
    def transform(self, df):
        pass