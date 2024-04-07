import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk


def pre_processing(data):
    stop_words = set(stopwords.words('english'))
    stop_words = stopwords.words('english')

    data['User_input_preprocessed'] = data['User_input']
    data['User_input_preprocessed'] = data['User_input_preprocessed'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    data['User_input_preprocessed'] = data['User_input_preprocessed'].apply(
        lambda x: ' '.join([word for word in x.split() if len(word) > 1]))
    data['User_input_preprocessed'] = data['User_input_preprocessed'].str.lower()
    data['User_input_preprocessed'] = data['User_input_preprocessed'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    data['User_input_preprocessed'] = data['User_input_preprocessed'].str.replace('\d+', '')

    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    data['User_input_preprocessed'] = data['User_input_preprocessed'].apply(lambda x: tokenizer.tokenize(x))
    data['User_input'] = data['User_input'].apply(lambda x: tokenizer.tokenize(x))


def stemming(data):
    stemmer = nltk.stem.PorterStemmer()
    data['User_input_preprocessed_stem'] = data['User_input_preprocessed'].apply(
        lambda x: [stemmer.stem(word) for word in x])


def lemmatization(data):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    data['User_input_preprocessed_lem'] = data['User_input_preprocessed'].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x])


def tf_stemming(data):
    pre_processing(data)
    stemming(data)
    tf = TfidfVectorizer()
    return tf.fit_transform(data['User_input_preprocessed_stem'].apply(' '.join))


def tf_lemmatization(data):
    pre_processing(data)
    lemmatization(data)
    tf = TfidfVectorizer()
    return tf.fit_transform(data['User_input_preprocessed_lem'].apply(' '.join))


# Testa aqui mariana
import pandas as pd
import joblib
model = joblib.load('Models/LR/LR_stem_tfidf.pkl')

def getDiagnosis(user_input):
    data = pd.DataFrame(columns=['User_input'])
    data.columns = data.columns.astype(str)
    data['User_input'] = [user_input]
    processed_data = tf_stemming(data)
    prediction = model.predict(processed_data)
    disease = prediction[0]
    print(disease)


getDiagnosis(
    'I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches.')
