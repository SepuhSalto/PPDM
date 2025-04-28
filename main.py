# main.py
import tokenization
import stopword_removal
import stemming
import training

def main():
    # Step 1: Clean the text (remove stopwords)
    stopword_removal.remove_stopwords('data.csv', 'cleaned_file.csv')

    # Step 2: Tokenize the cleaned data
    tokenization.tokenize_data('cleaned_file.csv', 'tokenized_file.csv')

    # Step 3: Apply stemming on the tokenized words
    stemming.stem_data('tokenized_file.csv', 'stemmed_file.csv')

    # Step 4: Train the model with the stemmed data
    training.train_model('stemmed_file.csv')

if __name__ == "__main__":
    main()
