import pandas as pd
from steps.html_removal import remove_html
from steps.lowercasing import lowercase
from steps.tokenization import tokenize
from steps.punctuation import remove_punctuation
from steps.stopwords import remove_stopwords
from steps.lemmatization import lemmatize


class PreprocessingPipeline:
    def __init__(
        self,
        remove_html: bool = True,
        lowercase: bool = True,
        tokenize: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True
    ):
        self.remove_html = remove_html
        self.lowercase = lowercase
        self.tokenize = tokenize
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

    def process(self, text: str) -> list:
        if self.remove_html:
            text = remove_html(text)
        if self.lowercase:
            text = lowercase(text)
        if self.tokenize:
            text = tokenize(text)
        if self.remove_punctuation:
            text = remove_punctuation(text)
        if self.remove_stopwords:
            text = remove_stopwords(text)
        if self.lemmatize:
            text = lemmatize(text)
        return text


if __name__ == "__main__":
    df = pd.read_csv('data/IMDB Dataset.csv')

    pipeline = PreprocessingPipeline()
    df['cleaned'] = df['review'].apply(pipeline.process)

    print("Original:", df['review'][1][:200])
    print("Cleaned: ", df['cleaned'][1][:10])


    """
    Original: A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece
    Cleaned:  ['wonderful', 'little', 'production', 'filming', 'technique', 'fashion', 'give', 'comforting', 'sometimes', 'discomforting']
    """