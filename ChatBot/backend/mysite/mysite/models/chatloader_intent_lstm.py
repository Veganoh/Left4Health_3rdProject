import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import os


def loadmodel():
    global model, tokenizer, label_encoder, max_sequence_length
    intent_classifier_lstm = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'intent_classifier_lstm.pkl')
    chatbot_intent_model_lstm = os.path.join(os.path.dirname(os.path.realpath(__file__)),'chatbot_intent_model_lstm.h5')
    # Load tokenizer, label encoder, and max sequence length from intent_classifier_lstm.pkl
    with open(intent_classifier_lstm, 'rb') as f:
        tokenizer, label_encoder, max_sequence_length = pickle.load(f)

    # Load the LSTM model from chatbot_intent_model_lstm.h5
    model = load_model(chatbot_intent_model_lstm)

    return tokenizer, label_encoder, max_sequence_length, model


# Loading the model, tokenizer, label encoder, and max sequence length
model, tokenizer, label_encoder, max_sequence_length = loadmodel()


# Function to predict intent
def predict_intent_lstm(text):
    # Tokenize the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    # Pad sequences
    padded_sequence = pad_sequences(text_sequence, maxlen=max_sequence_length, padding='post')
    # Make predictions
    predicted_probabilities = model.predict(padded_sequence)
    # Get the predicted class label
    predicted_class = np.argmax(predicted_probabilities)
    # Convert the predicted class label back to the original label
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_disease


