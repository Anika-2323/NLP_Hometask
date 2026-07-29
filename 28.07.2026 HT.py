import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download stopwords
nltk.download("stopwords")

# Load Dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[["v1", "v2"]]

# Rename columns
data.columns = ["label", "message"]

print("First 5 Records:")
print(data.head())

# Stopwords and Stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Preprocessing Function
def preprocess(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply preprocessing
data["clean_message"] = data["message"].apply(preprocess)

print("\nPreprocessed Data:")
print(data[["message", "clean_message"]].head())

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(data["clean_message"])
y = data["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# User Input
user_message = input("\nEnter an Email/SMS: ")

clean_message = preprocess(user_message)

message_vector = tfidf.transform([clean_message])

prediction = model.predict(message_vector)

print("\nPrediction:", prediction[0])
