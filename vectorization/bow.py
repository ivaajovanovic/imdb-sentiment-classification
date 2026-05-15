from sklearn.feature_extraction.text import CountVectorizer
from vectorization.base import BaseVectorizer

class BoWVectorizer(BaseVectorizer):
    def __init__(self, ngram_range: tuple = (1, 1)):
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)
    
    def _prepare_texts(self, df):
        return df['cleaned'].apply(lambda tokens: ' '.join(tokens))
    
    def fit_transform(self, df):
        texts = self._prepare_texts(df)
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, df):
        texts = self._prepare_texts(df)
        return self.vectorizer.transform(texts)