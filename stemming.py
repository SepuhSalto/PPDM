# stemming.py
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def stem_data(input_file, output_file):
    df = pd.read_csv(input_file)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def clean_and_stem(tokens):
        if isinstance(tokens, str):
            tokens = eval(tokens)
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        return str(stemmed_tokens)

    df['stemmed_words'] = df['tokenized_words'].apply(clean_and_stem)
    df.to_csv(output_file, index=False)
    print(f"Stemming done and saved to {output_file}")
