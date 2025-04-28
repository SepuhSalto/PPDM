import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

df = pd.read_csv('cleaned_file.csv')  

df['tokenized_words'] = df['cleaned_review'].apply(word_tokenize)

print(df[['cleaned_review', 'tokenized_words']].head())

df.to_csv('tokenized_file.csv', index=False)