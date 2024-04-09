import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove punctuation and special characters, and lowercase the text
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token.lower()) for token in tokens]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)


# Load the dataset
data = pd.read_csv('dataset_full.csv')

# Preprocess the questions
data['Question'] = data['Question'].apply(preprocess_text)

# Save preprocessed data
data.to_csv('preprocessed_data.csv', index=False)
