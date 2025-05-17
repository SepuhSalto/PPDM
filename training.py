import pandas as pd
import math
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report  # Import classification_report
import pickle

# Custom Train-Test Split Function
def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    indices = np.random.permutation(num_samples)

    X_train = X[indices[:-num_test_samples]]
    y_train = y[indices[:-num_test_samples]]
    X_test = X[indices[-num_test_samples:]]
    y_test = y[indices[-num_test_samples:]]

    return X_train, X_test, y_train, y_test

def compute_tfidf(corpus, vocabulary=None):
    # If no vocabulary is provided (during prediction), create a vocabulary based on the corpus
    if vocabulary is None:
        all_words = set()
        for doc in corpus:
            words = doc.split()
            all_words.update(words)
        vocabulary = list(all_words)
    
    # Calculate Term Frequency (TF)
    tf = []
    for doc in corpus:
        word_count = Counter(doc.split())
        tf.append({word: count / len(doc.split()) for word, count in word_count.items()})
    
    # Calculate Inverse Document Frequency (IDF)
    num_docs = len(corpus)
    idf = {}
    for word in vocabulary:
        doc_count = sum(1 for doc in corpus if word in doc.split())
        idf[word] = math.log(num_docs / (1 + doc_count))  # Add 1 to avoid division by zero
    
    # Calculate TF-IDF
    tfidf = []
    for doc in tf:
        doc_tfidf = {word: tf_val * idf.get(word, 0) for word, tf_val in doc.items()}
        doc_vector = [doc_tfidf.get(word, 0) for word in vocabulary]  # Ensure the vector has the same length for all docs
        tfidf.append(doc_vector)
    
    # Convert to a numpy array to ensure it's 2D (n_samples, n_features)
    return np.array(tfidf), vocabulary


# Custom SVM Implementation (Linear SVM)
class CustomSVM:
    def __init__(self, learning_rate=0.0001, reg_strength=1, num_iters=1000):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.num_iters = num_iters
        self.class_map = {}  # To store the mapping from numeric labels to string labels
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_map = {i: label for i, label in enumerate(self.classes)}  # Create mapping from numeric to string
        num_classes = len(self.classes)
        y_encoded = np.array([np.where(self.classes == label)[0][0] for label in y])
        
        num_samples, num_features = X.shape
        self.W = np.random.randn(num_features, num_classes) * 0.01  # Weights initialization
        
        for _ in range(self.num_iters):
            scores = np.dot(X, self.W)
            correct_class_scores = scores[np.arange(num_samples), y_encoded]
            margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)
            margins[np.arange(num_samples), y_encoded] = 0  # Correct class margin is 0
            
            loss = np.sum(margins) / num_samples + 0.5 * self.reg_strength * np.sum(self.W * self.W)
            margin_mask = margins > 0
            margin_mask[np.arange(num_samples), y_encoded] = -np.sum(margin_mask, axis=1)
            dW = np.dot(X.T, margin_mask) / num_samples + self.reg_strength * self.W
            self.W -= self.learning_rate * dW
            
        return self
    
    def predict(self, X):
        scores = np.dot(X, self.W)
        y_pred_numeric = np.argmax(scores, axis=1)
        y_pred_string = [self.class_map[label] for label in y_pred_numeric]  # Map numeric predictions to string
        return y_pred_string

# Grid Search Implementation
def grid_search(input_file):
    # Load the data
    df = pd.read_csv(input_file)
    
    # Combine `category` and `sentiment` into a single column `category_sentiment`
    df['category_sentiment'] = df['category'] + '-' + df['sentiment']
    
    reviews = df['stemmed_words']
    tfidf_reviews, all_words = compute_tfidf(reviews)  # Vectorized documents
    
    # Use `category_sentiment` as the target
    y = df['category_sentiment']

    # Hyperparameter grid to search over
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    reg_strengths = [0.1, 1, 10]
    num_iters_list = [500, 1000, 2000]

    best_accuracy = 0
    best_params = {}

    # Perform grid search
    for learning_rate in learning_rates:
        for reg_strength in reg_strengths:
            for num_iters in num_iters_list:
                # Split data
                X_train, X_test, y_train, y_test = manual_train_test_split(np.array(tfidf_reviews), np.array(y), test_size=0.2, random_state=42)
                
                # Train the model with the current hyperparameters
                svm = CustomSVM(learning_rate=learning_rate, reg_strength=reg_strength, num_iters=num_iters)
                svm.fit(X_train, y_train)
                
                # Get predictions
                predictions = svm.predict(X_test)
                
                # Calculate accuracy (you can use other evaluation metrics like F1-score if preferred)
                accuracy = np.mean(predictions == y_test)
                
                print(f"Trying learning_rate={learning_rate}, reg_strength={reg_strength}, num_iters={num_iters} -> Accuracy: {accuracy}")
                
                # Update best model if this combination is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'learning_rate': learning_rate,
                        'reg_strength': reg_strength,
                        'num_iters': num_iters
                    }
    
    print("\nBest Hyperparameters:")
    print(best_params)
    print(f"Best Accuracy: {best_accuracy}")

# Final train_model function using the best hyperparameters found
def train_model(input_file):
    # Load the data
    df = pd.read_csv(input_file)
    
    # Combine `category` and `sentiment` into a single column `category_sentiment`
    df['category_sentiment'] = df['category'] + '-' + df['sentiment']
    
    reviews = df['stemmed_words']
    y = df['category_sentiment']

    # Use custom compute_tfidf function to compute TF-IDF values
    X_tfidf, vocabulary = compute_tfidf(reviews)
    
    # Manually split the data into training and testing sets
    X_train, X_test, y_train, y_test = manual_train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Train the custom SVM with predefined hyperparameters
    best_learning_rate = 0.0001
    best_reg_strength = 0.1
    best_num_iters = 2000
    
    svm = CustomSVM(learning_rate=best_learning_rate, reg_strength=best_reg_strength, num_iters=best_num_iters)
    svm.fit(X_train, y_train)
    
    # Save the trained model
    with open('svm_model.pkl', 'wb') as model_file:
        pickle.dump(svm, model_file)

    # Save the vocabulary (for future predictions)
    with open('vocabulary.pkl', 'wb') as vocab_file:
        pickle.dump(vocabulary, vocab_file)
    
    # Make predictions (for evaluation)
    predictions = svm.predict(X_test)
    
    # Evaluate model performance and print classification report
    print("Classification Report:\n")
    print(classification_report(y_test, predictions))