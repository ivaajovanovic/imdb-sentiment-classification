from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

def remove_stopwords(tokens: list) -> list:
    return [token for token in tokens if token not in STOP_WORDS]