from abc import ABC, abstractmethod

class PreprocessingStep(ABC):
    @abstractmethod
    def run(self, text):
        pass