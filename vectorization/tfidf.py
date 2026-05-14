from sklearn.feature_extraction.text import TfidfVectorizer

import os
import pickle

def get_tfidf_vectorizer(ngram_range: tuple = (1, 1)):
    return TfidfVectorizer(ngram_range=ngram_range)

def vectorize(df, vectorizer):
   
    texts = df['cleaned'].apply(lambda tokens: ' '.join(tokens))


    X = vectorizer.fit_transform(texts)

    # vrednosti su decimalni brojevi (0.0 - 1.0) umesto celih (bow)

    #with open("results/tfidf_vectors.pkl", "wb") as f:
    #    pickle.dump(X, f)

    return X
