# stopword_removal.py
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def remove_stopwords(input_file, output_file):
    df = pd.read_csv(input_file)
    stop_words = set(stopwords.words('indonesian'))

    def clean_text(text):
        text = text.lower()
        valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
        text = ''.join([char for char in text if char in valid_chars])
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)

    df['cleaned_review'] = df['review'].apply(clean_text)
    df.to_csv(output_file, index=False)
    print(f"Stopwords removed and saved to {output_file}")
