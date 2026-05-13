from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def lemmatize(tokens: list) -> list:
    return [lemmatizer.lemmatize(token) for token in tokens]