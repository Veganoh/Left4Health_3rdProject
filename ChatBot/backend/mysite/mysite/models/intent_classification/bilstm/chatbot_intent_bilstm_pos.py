import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import os
# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# Function to tokenize sentences and perform POS tagging


def tokenize_and_pos_tag(sentences):
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]  # Lowercasing for consistency
    pos_tags = [pos_tag(tokens) for tokens in tokenized_sentences]
    return pos_tags


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up
two_levels_up = os.path.abspath(os.path.join(current_dir, '..', '..'))
dataset_file_path = os.path.join(two_levels_up, 'dataset_full.csv')
# Load the dataset
df = pd.read_csv(dataset_file_path)
# Tokenize sentences and perform POS tagging
pos_tags = tokenize_and_pos_tag(df['Question'].values)
# Create features based on POS tags (if you want to use POS as features)


def train_intent_bilstm_pos():
    global pos_tags
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Question'])
    X_sequences = tokenizer.texts_to_sequences(df['Question'])
    # Pad sequences to ensure uniform length
    max_sequence_length = max([len(seq) for seq in X_sequences])
    X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length, padding='post')
    # Encode the 'Disease' column
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['Disease'])
    # Split data into train and test sets
    X_train_padded, X_test_padded, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)
    # Define BiLSTM model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, name='intent_classifier_bilstm_embedding', output_dim=128, input_shape=(max_sequence_length,)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, name='lstm')))
    model.add(Dense(len(label_encoder.classes_), activation='softmax', name='dense'))  # Number of classes
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train model
    model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=10, batch_size=32, verbose=1)
    # Save model and tokenizer
    model.save('intent_classifier_bilstm.keras')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    # Save the label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)


def predict_intent_bilstm_pos(question):
    # Load the trained model
    model = load_model('intent_classifier_bilstm.keras')

    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Tokenize and convert to sequence
    sequence = tokenizer.texts_to_sequences([question])

    # Pad the sequence
    max_sequence_length = model.input_shape[1]  # Assuming the model has only one input
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')

    # Perform the prediction
    prediction = model.predict(padded_sequence)

    # Load the label encoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Convert prediction to label
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    return predicted_label[0]  # Return the predicted label as a string

