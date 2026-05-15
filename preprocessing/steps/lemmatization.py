from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet', quiet=True)

from preprocessing.steps.base import PreprocessingStep

class LemmatizationStep(PreprocessingStep):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def run(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(token) for token in tokens]