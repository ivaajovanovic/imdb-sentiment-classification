import pandas as pd
import os

from preprocessing.steps.html_removal import HtmlRemovalStep
from preprocessing.steps.lowercasing import LowercasingStep
from preprocessing.steps.tokenization import TokenizationStep
from preprocessing.steps.punctuation import PunctuationRemovalStep
from preprocessing.steps.stopwords import StopwordsRemovalStep
from preprocessing.steps.lemmatization import LemmatizationStep


class PreprocessingPipeline:
    def __init__(self, steps: list):
        self.steps = steps

    def process(self, text):
        for step in self.steps:
            text = step.run(text)
        return text


def get_default_pipeline():
    return PreprocessingPipeline(steps=[
        HtmlRemovalStep(),
        LowercasingStep(),
        TokenizationStep(),
        PunctuationRemovalStep(),
        StopwordsRemovalStep(),
        LemmatizationStep(),
    ])
