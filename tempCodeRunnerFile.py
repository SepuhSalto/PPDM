import tokenization
import stopword_removal
import stemming
import training

def main():
    # Preprocessing steps
    #stopword_removal.remove_stopwords('data.csv', 'cleaned_file.csv')
    #tokenization.tokenize_data('cleaned_file.csv', 'tokenized_file.csv')
    #stemming.stem_data('tokenized_file.csv', 'stemmed_file.csv')

    # Training the model
    training.train_model('stemmed_file.csv')

if __name__ == "__main__":
    main()
