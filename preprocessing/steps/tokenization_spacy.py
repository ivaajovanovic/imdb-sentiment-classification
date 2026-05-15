import spacy
from steps.base import PreprocessingStep

class TokenizationSpacyStep(PreprocessingStep):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def run(self, text: str) -> list:
        doc = self.nlp(text)
        return [token.text for token in doc]