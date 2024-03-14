import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Load the CSV file into a DataFrame
df = pd.read_csv('sampleDataset.csv')

# Assuming 'text', 'label', and 'speech' are the column names in your CSV file
texts = df['text']
disease_labels = df['label']
speech_labels = df['speech']

# Split dataset into train and test sets
X_train, X_test, y_train_disease, y_test_disease, y_train_speech, y_test_speech = train_test_split(texts, disease_labels, speech_labels, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# Pad sequences (optional if using fixed input length)
maxlen = max([len(x.split()) for x in texts])
X_train_padded = pad_sequences(X_train_tfidf, maxlen=maxlen)
X_test_padded = pad_sequences(X_test_tfidf, maxlen=maxlen)

# Convert disease labels to one-hot encoding
num_classes_disease = len(df['label'].unique())
y_train_disease_encoded = to_categorical(y_train_disease, num_classes=num_classes_disease)
y_test_disease_encoded = to_categorical(y_test_disease, num_classes=num_classes_disease)

# Convert speech labels to one-hot encoding
num_classes_speech = len(df['speech'].unique())
y_train_speech_encoded = to_categorical(y_train_speech, num_classes=num_classes_speech)
y_test_speech_encoded = to_categorical(y_test_speech, num_classes=num_classes_speech)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tfidf_vectorizer.get_feature_names_out()), output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes_disease, activation='softmax'))  # Output layer for disease classification

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for disease classification
model.fit(X_train_padded, y_train_disease_encoded, epochs=3, batch_size=32, validation_data=(X_test_padded, y_test_disease_encoded))

# Evaluate the model for disease classification
loss, accuracy = model.evaluate(X_test_padded, y_test_disease_encoded)
print("Disease Classification Test Accuracy:", accuracy)

# Output layer for speech classification
model.add(Dense(num_classes_speech, activation='softmax'))

# Compile the model for speech classification
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for speech classification
model.fit(X_train_padded, y_train_speech_encoded, epochs=5, batch_size=32, validation_data=(X_test_padded, y_test_speech_encoded))

# Evaluate the model for speech classification
loss, accuracy = model.evaluate(X_test_padded, y_test_speech_encoded)
print("Speech Classification Test Accuracy:", accuracy)
