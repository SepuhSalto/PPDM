import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def tokenize_data(input_file, output_file):
    df = pd.read_csv(input_file)  
    df['tokenized_words'] = df['cleaned_review'].apply(word_tokenize)
    df.to_csv(output_file, index=False)
    print(f"Tokenization done and saved to {output_file}")