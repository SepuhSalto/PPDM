import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if you haven't already
nltk.download('stopwords')

# Step 1: Load the CSV file (assuming your CSV file has a 'review' column containing text data)
df = pd.read_csv('data.csv')  # Replace 'your_file.csv' with your actual CSV file name

# Step 2: Get stopwords (for Indonesian in this case)
stop_words = set(stopwords.words('indonesian'))  # You can change to 'english' for English text

# Step 3: Function to clean text by removing special characters and stopwords
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters (only keep alphabets and spaces)
    valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    text = ''.join([char for char in text if char in valid_chars])
    
    # Split the text into words
    words = text.split()
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join the filtered words back into a single string
    cleaned_text = ' '.join(filtered_words)
    
    return cleaned_text

# Step 4: Apply the function to the 'review' column (or whatever column contains the text data)
df['cleaned_review'] = df['review'].apply(clean_text)

# Step 5: (Optional) Save the cleaned data back into a CSV file
df.to_csv('cleaned_file.csv', index=False)

# Display the cleaned data (for verification)
print(df[['review', 'cleaned_review']].head())
