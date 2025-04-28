import pandas as pd
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')

df = pd.read_csv('tokenized_file.csv')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_and_stem(tokens):
    if isinstance(tokens, str):
        tokens = eval(tokens)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return str(stemmed_tokens)

df['stemmed_words'] = df['tokenized_words'].apply(clean_and_stem)

print(df[['tokenized_words', 'stemmed_words']].head())

df.to_csv('stemmed_file.csv', index=False)
