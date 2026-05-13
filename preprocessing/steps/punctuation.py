def remove_punctuation(tokens: list) -> list:
    return [token for token in tokens if token.isalpha()]