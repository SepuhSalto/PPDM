import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Step 1: Load the CSV file into a pandas DataFrame
df = pd.read_csv('stemmed_file.csv')  # Replace 'stemmed_file.csv' with your file path

# Step 2: Extract the features (stemmed words) and labels (sentiment)
# Assuming 'stemmed_words' column contains the stemmed text as a single string
X = df['stemmed_words']  # The feature (text data)
y = df['sentiment']      # The labels (sentiment: positive/negative)

# Step 3: Convert the text data into TF-IDF features using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)  # Transform the text data into TF-IDF features

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 5: Train a model (Support Vector Machine in this case)
model = SVC(kernel='linear')  
model.fit(X_train, y_train)

# Step 6: Predict on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print(classification_report(y_test, y_pred))
print(y_pred[:10]) 
