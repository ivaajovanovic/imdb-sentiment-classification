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


if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.path.append('preprocessing')
    from pipeline import PreprocessingPipeline

    df = pd.read_csv('data/IMDB Dataset.csv')

    pipeline = PreprocessingPipeline()
    df['cleaned'] = df['review'].apply(pipeline.process)

    vectorizer = get_tfidf_vectorizer()
    X = vectorize(df, vectorizer)

    print(f"Shape: {X.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"\nPrimer vektora za prvu recenziju (samo nenulte vrednosti):")

    first_review = X[0]
    nonzero_indices = first_review.nonzero()[1]
    for idx in nonzero_indices[:10]:
        word = list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(idx)]
        print(f"  '{word}': {first_review[0, idx]:.4f}")