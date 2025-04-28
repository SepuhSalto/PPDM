import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK datasets
nltk.download('punkt')

# Step 1: Load the cleaned CSV file into a pandas DataFrame
df = pd.read_csv('cleaned_file.csv')  # Replace 'cleaned_file.csv' with the actual file path

# Step 2: Tokenize the 'cleaned_review' column (assuming the cleaned text is in this column)
df['tokenized_words'] = df['cleaned_review'].apply(word_tokenize)

# Step 3: Print the tokenized words for the first few rows to verify
print(df[['cleaned_review', 'tokenized_words']].head())

# Optional: Save the tokenized data back to a new CSV file
df.to_csv('tokenized_file.csv', index=False)