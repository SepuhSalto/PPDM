import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv('data.csv')  
stop_words = set(stopwords.words('indonesian'))  

def clean_text(text):
    text = text.lower()
    valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    text = ''.join([char for char in text if char in valid_chars])
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    
    return cleaned_text

df['cleaned_review'] = df['review'].apply(clean_text)

df.to_csv('cleaned_file.csv', index=False)

print(df[['review', 'cleaned_review']].head())
