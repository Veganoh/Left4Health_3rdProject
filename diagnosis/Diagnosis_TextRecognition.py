import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Sample dataset (replace with your data)
data = pd.DataFrame({
    'text': ['This is a sample text for disease 0.', 'Another sample text for disease 1.', 'Yet another example for disease 2.', 'Text for disease 3.', 'Description for disease 4.'],
    'label': [0, 1, 2, 3, 4]
})

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# Pad sequences (optional if using fixed input length)
maxlen = max([len(x.split()) for x in data['text']])
X_train_padded = pad_sequences(X_train_tfidf, maxlen=maxlen)
X_test_padded = pad_sequences(X_test_tfidf, maxlen=maxlen)

# Convert labels to one-hot encoding
num_classes = len(data['label'].unique())
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tfidf_vectorizer.get_feature_names_out()), output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, y_train_encoded, epochs=5, batch_size=32, validation_data=(X_test_padded, y_test_encoded))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)
print("Test Accuracy:", accuracy)
