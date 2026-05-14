from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

def get_bow_vectorizer(ngram_range: tuple = (1, 1)): #testirati sa razlicitim kombinacijama n-grama
    return CountVectorizer(ngram_range=ngram_range)

def vectorize(df, vectorizer):
    # tokeni -> string jer bow ocekuje stringove
    texts = df['cleaned'].apply(lambda tokens: ' '.join(tokens))

    # fit pravi recnik jedinstvenih reci
    # transform pretvara svaki red- recenziju u vektor duzine erecnika
    X = vectorizer.fit_transform(texts)

    #os.makedirs("results", exist_ok=True)
    #with open("results/bow_vectors.pkl", "wb") as f:
    #    pickle.dump(X, f)

    return X