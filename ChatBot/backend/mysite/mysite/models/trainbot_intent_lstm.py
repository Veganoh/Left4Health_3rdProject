import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
import os


def save_model(model, tokenizer, label_encoder, max_sequence_length):
    with open('intent_classifier_lstm.pkl', 'wb') as f:
        pickle.dump((tokenizer, label_encoder, max_sequence_length), f)
    print('Done dumping into pickle')
    model.save('chatbot_intent_model_lstm.h5')
    print('Saved model')


def train_model_intent_lstm(request):
    # Read the CSV file into a pandas DataFrame
    dataset_full = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_full.csv')
    df = pd.read_csv(dataset_full)

    # Concatenate question and answer
    X = df['Question'] + ' ' + df['Answer']
    y = df['Disease']

    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)

    # Pad sequences to ensure uniform length
    max_sequence_length = max([len(x) for x in X])
    X = pad_sequences(X, maxlen=max_sequence_length, padding='post')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    # Save the model, tokenizer, label encoder, and max sequence length
    save_model(model, tokenizer, label_encoder, max_sequence_length)

