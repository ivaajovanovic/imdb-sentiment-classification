from sklearn.feature_extraction.text import CountVectorizer

def get_bow_vectorizer(ngram_range: tuple = (1, 1)): #testirati sa razlicitim kombinacijama n-grama
    return CountVectorizer(ngram_range=ngram_range)

def vectorize(df, vectorizer):
    # Spajamo tokene nazad u string jer sklearn ocekuje string
    texts = df['cleaned'].apply(lambda tokens: ' '.join(tokens))
    # fit pravi recnik jedinstvenih reci
    # transform pretvara svaki red- recenziju u vektor duzine erecnika
    X = vectorizer.fit_transform(texts)
    # X - matrica, redovi su recenzije a kolone jedinstvene reci

    return X