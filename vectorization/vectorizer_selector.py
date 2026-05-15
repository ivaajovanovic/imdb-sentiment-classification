from vectorization.bow import BoWVectorizer
from vectorization.tfidf import TfidfVectorizerWrapper

class VectorizerSelector:
    def __init__(self, method: str = 'tfidf', ngram_range: tuple = (1, 1)):
        self.method = method
        self.ngram_range = ngram_range
    
    def get_vectorizer(self):
        if self.method == 'bow':
            return BoWVectorizer(ngram_range=self.ngram_range)
        elif self.method == 'tfidf':
            return TfidfVectorizerWrapper(ngram_range=self.ngram_range)
        else:
            raise ValueError(f"Unknown method: {self.method}")