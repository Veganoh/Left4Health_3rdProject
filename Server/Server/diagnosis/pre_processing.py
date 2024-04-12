import re
from nltk.corpus import stopwords
import joblib
import pandas as pd
import os
import nltk


nltk.download('stopwords')


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models/model_text.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer')


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
    data['User_input_preprocessed'] = data['User_input_preprocessed'].str.replace(r'\d+', '')

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
    tf = joblib.load(vectorizer_path + '/tfidf_stem.pkl')
    return tf.transform(data['User_input_preprocessed_stem'].apply(' '.join))


def tf_lemmatization(data):
    pre_processing(data)
    lemmatization(data)
    tf = joblib.load(vectorizer_path + '/tfidf_lem.pkl')
    return tf.transform(data['User_input_preprocessed_lem'].apply(' '.join))


#model = joblib.load(model_path)


def runModel(user_input):
    data = pd.DataFrame({'User_input': [user_input]})
    processed_data = tf_stemming(data)
    prediction = model.predict(processed_data)
    disease = prediction[0]
    return disease


import keras
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models\model_text.keras')
print(os.path.isfile(model_path))


model = keras.models.load_model(model_path)