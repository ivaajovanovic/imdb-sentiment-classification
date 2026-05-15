from vectorization.bow import BoWVectorizer
from vectorization.tfidf import TfidfVectorizerWrapper

class VectorizerSelector:
    def __init__(self, method: str = 'tfidf', ngram_range: tuple = (1, 1), min_df: int = 1):
        self.method = method
        self.ngram_range = ngram_range
        self.min_df = min_df
    
    def get_vectorizer(self):
        if self.method == 'bow':
            return BoWVectorizer(ngram_range=self.ngram_range, min_df=self.min_df)
        elif self.method == 'tfidf':
            return TfidfVectorizerWrapper(ngram_range=self.ngram_range, min_df=self.min_df)