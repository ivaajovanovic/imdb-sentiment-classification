from preprocessing.steps.base import PreprocessingStep

class PunctuationRemovalStep(PreprocessingStep):
    def run(self, tokens: list) -> list:
        return [token for token in tokens if token.isalpha()]