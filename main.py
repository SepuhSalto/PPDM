import streamlit as st
import pickle
import numpy as np
from training import train_model
from training import compute_tfidf  # Import your custom compute_tfidf function
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download necessary NLTK data if not already available
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the input text (tokenization, stopword removal, and stemming)
def preprocess_input_text(input_text):
    # 1. Tokenize the input text
    tokens = word_tokenize(input_text.lower())
    
    # 2. Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 3. Stem the filtered tokens
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # 4. Convert the stemmed tokens back to a string
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text

# Function to predict a single input text
def predict_single(input_text):
    # Load the saved vocabulary
    with open('vocabulary.pkl', 'rb') as vocab_file:
        vocabulary = pickle.load(vocab_file)
    
    # Preprocess the input text
    preprocessed_text = preprocess_input_text(input_text)
    
    # Compute the TF-IDF values for the preprocessed text using the saved vocabulary
    tfidf_input, _ = compute_tfidf([preprocessed_text], vocabulary)  # This returns a 2D array
    
    # Load the trained model
    with open('svm_model.pkl', 'rb') as model_file:
        svm = pickle.load(model_file)
    
    # Predict the category of the input text
    prediction = svm.predict(tfidf_input)
    
    return prediction[0]  # Return the predicted category


# Streamlit UI
def main():
    st.title('Text Category Prediction using Custom SVM')

    # Input text field for new text
    input_text = st.text_area("Enter your text for prediction:", "")
    
    if st.button("Train Model"):
        # Train the model (this will store the trained model and vocabulary)
        train_model('stemmed_file.csv')
        st.write("Model has been trained successfully!")

    if st.button("Predict"):
        if input_text.strip() != "":
            # Predict the category for the input text using the trained model
            prediction = predict_single(input_text)
            st.write(f"The predicted category for the input text is: **{prediction}**")
        else:
            st.error("Please enter some text to classify.")

if __name__ == "__main__":
    main()
