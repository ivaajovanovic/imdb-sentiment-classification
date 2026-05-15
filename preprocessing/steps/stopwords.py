from nltk.corpus import stopwords
from preprocessing.steps.base import PreprocessingStep

class StopwordsRemovalStep(PreprocessingStep):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def run(self, tokens: list) -> list:
        return [token for token in tokens if token not in self.stop_words]