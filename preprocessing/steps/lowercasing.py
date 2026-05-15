from preprocessing.steps.base import PreprocessingStep

class LowercasingStep(PreprocessingStep):
    def run(self, text: str) -> str:
        return text.lower()