import numpy as np
import pandas as pd
import re
from keras.models import load_model
import pickle
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load files:
variables_preprocessed = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'variables_preprocessed.pkl')
tokenizer_preprocessed = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tokenizer_preprocessed.pkl')
chatbot_model_preprocessed = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chatbot_model_preprocessed.h5')


# Load the saved variables
with open(variables_preprocessed, 'rb') as f:
    max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, target_token_index = pickle.load(f)

reverse_target_char_index = {index: char for char, index in target_token_index.items()}

# Load the saved tokenizer
with open(tokenizer_preprocessed, 'rb') as f:
    input_token_index, target_token_index = pickle.load(f)

# Load the pre-trained chatbot model
model = load_model(chatbot_model_preprocessed)


# Define a function to preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Transform to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def find_key(dictionary, value):
    keys = [key for key, val in dictionary.items() if val == value]
    return keys


# Define a function to generate responses
def generate_response(input_text):
    input_text = preprocess_text(input_text)
    input_sequence = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(input_text):
        if char in input_token_index:
            input_sequence[0, t, input_token_index[char]] = 1.0

    decoder_input = np.zeros((1, max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    generated_response = ''
    for i in range(1, max_decoder_seq_length):
        decoder_output = model.predict([input_sequence, decoder_input])
        print(i)
        sampled_token_index = np.argmax(decoder_output[0, i - 1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        generated_response += sampled_char
        if sampled_char == '\n':
            break
        decoder_input[0, i, sampled_token_index] = 1.0

    return generated_response

def generate_intent_svc(text):
    model = joblib.load('intent_classifier_model_svc.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    # Use the model for prediction
    vtext = vectorizer.transform([text])
    predicted_intent_proba = model.predict_proba(vtext)
    predicted_intent_proba = predicted_intent_proba[0]
    print(predicted_intent_proba)
    classes = model.classes_
    predicted_intent = []
    for class_name, proba in zip(classes, predicted_intent_proba):
        print(f"{class_name}: {proba}")
        predicted_intent.append({'disease': class_name, 'probability': proba})

    return predicted_intent


# Example usage
user_query = "What are some common treatments for Psoriasis?"
response = generate_response(user_query)
print("User Query:", user_query)
print("Chatbot Response:", response)
