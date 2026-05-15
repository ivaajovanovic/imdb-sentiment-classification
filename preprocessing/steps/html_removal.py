import re
from preprocessing.steps.base import PreprocessingStep

class HtmlRemovalStep(PreprocessingStep):
    def run(self, text: str) -> str:
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text