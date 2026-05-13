from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(ngram_range: tuple = (1, 1)):
    return TfidfVectorizer(ngram_range=ngram_range)

def vectorize(df, vectorizer):
    # Spajamo tokene nazad u string jer sklearn ocekuje string
    texts = df['cleaned'].apply(lambda tokens: ' '.join(tokens))
    # fit pravi recnik i racuna IDF vrednosti za svaku rec
    # transform pretvara svaku recenziju u vektor TF-IDF vrednosti
    X = vectorizer.fit_transform(texts)
    # X - matrica, redovi su recenzije a kolone jedinstvene reci
    # vrednosti su decimalni brojevi (0.0 - 1.0) umesto celih
    return X
