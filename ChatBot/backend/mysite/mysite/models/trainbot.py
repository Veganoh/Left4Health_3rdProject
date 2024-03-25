import os

import numpy as np
import pandas as pd
import re
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset from CSV file
dataset_full = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_full.csv')
dataset = pd.read_csv(dataset_full)


# Preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Transform to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def train():

    dataset['Question'] = dataset['Question'].apply(preprocess_text)
    dataset['Answer'] = dataset['Answer'].apply(preprocess_text)

    # Remove duplicates
    dataset.drop_duplicates(inplace=True)

    # Extract input texts and target texts
    input_texts = dataset['Question'].values
    target_texts = dataset['Answer'].values

    # Tokenize the input and target sequences
    input_characters = sorted(set(' '.join(input_texts)))
    target_characters = sorted(set(' '.join(target_texts)))

    input_token_index = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    # Prepare data for training
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

    # Define the model
    latent_dim = 256

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=1,
              epochs=25,
              validation_split=0.2)

    # Save the model
    model.save('chatbot_model_preprocessed.h5')

    # Save the tokenizer
    with open('tokenizer_preprocessed.pkl', 'wb') as f:
        pickle.dump((input_token_index, target_token_index), f)

    # Save variables into a file
    with open('variables_preprocessed.pkl', 'wb') as f:
        pickle.dump((max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, target_token_index), f)

    return True