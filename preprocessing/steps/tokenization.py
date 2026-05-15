from nltk.tokenize import word_tokenize
from preprocessing.steps.base import PreprocessingStep

class TokenizationStep(PreprocessingStep):
    def run(self, text: str) -> list:
        return word_tokenize(text)