import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib


# Load the dataset
dataset_full = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_full.csv')
data = pd.read_csv(dataset_full)
#model = joblib.load('intent_classifier_model.pkl')
# Preprocessing (if needed)


def train_model_intent():
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
    X = vectorizer.fit_transform(data['Question']  + ' ' + data['Answer'])  # Concatenate question and answer)
    y = data['Disease']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection and training
    model = SVC(probability=True, kernel='linear')  # Support Vector Machine with linear kernel
    model.fit(X_train, y_train)
    joblib.dump(model, 'intent_classifier_model_svc.pkl')
    vectorizer.fit(data['Question'])
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    # Evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

