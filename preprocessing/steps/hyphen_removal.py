import re
from preprocessing.steps.base import PreprocessingStep

class HyphenRemovalStep(PreprocessingStep):
    def run(self, text: str) -> str:
        return re.sub(r'-+', ' ', text)