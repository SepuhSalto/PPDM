import pandas as pd
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download necessary NLTK datasets if not already downloaded
nltk.download('punkt')

# Step 1: Load the tokenized CSV file into a pandas DataFrame
df = pd.read_csv('tokenized_file.csv')  # Replace 'tokenized_file.csv' with the actual path

# Initialize the Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Step 2: Function to clean tokenized words and apply stemming
def clean_and_stem(tokens):
    # Ensure that the tokens are properly split into words (and not characters)
    if isinstance(tokens, str):
        tokens = eval(tokens)  # Convert string representation of list to actual list
    # Apply stemming to each token (word)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return str(stemmed_tokens)  # Convert the list of stemmed words to a string

# Step 3: Apply the clean_and_stem function to the tokenized_words column
df['stemmed_words'] = df['tokenized_words'].apply(clean_and_stem)

# Step 4: Print the original tokenized words and the stemmed words for the first few rows to verify
print(df[['tokenized_words', 'stemmed_words']].head())

# Optional: Save the stemmed data back into a new CSV file
df.to_csv('stemmed_file.csv', index=False)
