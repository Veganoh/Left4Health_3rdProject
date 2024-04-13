import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features
from transformers import InputExample
import tensorflow as tf
import os
import pickle
import nltk

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up
two_levels_up = os.path.abspath(os.path.join(current_dir, '..', '..'))
dataset_file_path = os.path.join(two_levels_up, 'dataset_with_intents.csv')
# Load the dataset
df = pd.read_csv(dataset_file_path)
# Drop rows with missing values in any column
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)
# Preprocess the dataset
# Assume 'Question' column for the input and 'Disease' column for the intent labels
questions = df['Question'].tolist()
answers = df['Answer'].tolist()  # If you want to include answers in the model
disease_intents = df['Intent'].tolist()
# Combine questions and answers into a single text input
combined_texts = [question + " [SEP] " + answer for question, answer in zip(questions, answers)]
# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(disease_intents)

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    combined_texts, labels, test_size=0.2, random_state=42
)


# Tokenize and encode sentences in BERT's format
def encode_sentences(sentences, max_length=128):
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attention masks.
            return_tensors='tf',  # Return tensorflow tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = tf.concat(input_ids, 0)
    attention_masks = tf.concat(attention_masks, 0)
    return input_ids, attention_masks


# Tokenize and encode sentences for training and testing sets
train_input_ids, train_attention_masks = encode_sentences(train_texts)
test_input_ids, test_attention_masks = encode_sentences(test_texts)

# Convert labels to tensorflow tensors
train_labels = tf.convert_to_tensor(train_labels)
test_labels = tf.convert_to_tensor(test_labels)


# Create a TensorFlow dataset
def create_tf_dataset(input_ids, attention_masks, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(({
                                                      'input_ids': input_ids,
                                                      'attention_mask': attention_masks
                                                  }, labels))
    dataset = dataset.shuffle(len(input_ids)).batch(batch_size)
    return dataset


# Create the TensorFlow datasets
train_dataset = create_tf_dataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = create_tf_dataset(test_input_ids, test_attention_masks, test_labels)

def train_intent_bart():
    global labels
    input_ids, attention_masks = encode_sentences(combined_texts)
    # Convert labels to tensorflow tensors
    labels = tf.convert_to_tensor(labels)

    # Create the dataset
    dataset = create_tf_dataset(input_ids, attention_masks, labels)
    # Load the BERT model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
    # Prepare the model for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train the model
    model.fit(dataset, epochs=8)
    # Save the fine-tuned model and the label encoder for later use
    label_encoder_path = os.path.join(current_dir, 'model/label_encoder_v1.pkl')
    my_fine_tuned_bert = os.path.join(current_dir, 'model/my_fine_tuned_bert_v1')
    model.save_pretrained(my_fine_tuned_bert)

    with open(label_encoder_path, 'wb') as le_file:
        pickle.dump(label_encoder, le_file)

    # Evaluate the model on the test set
    result = model.evaluate(test_dataset)
    print(f"Test Loss: {result[0]}, Test Accuracy: {result[1]}")

    # Get predictions for the test set
    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions.logits, axis=1)

    try:
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(test_labels, predicted_labels)
        print(conf_matrix)
    except ValueError:
        print('failed to generate confusion matrix')
    # Calculate the classification report for each class
    try:
        # Get the unique labels from the true and predicted labels
        unique_labels = np.unique(np.concatenate((test_labels, predicted_labels)))

        # Calculate the classification report for each class that is present in the true or predicted labels
        class_report = classification_report(test_labels, predicted_labels, labels=unique_labels,
                                             target_names=label_encoder.inverse_transform(unique_labels))
        print(class_report)
    except ValueError:
        print('failed to generate classification report')

    try:
        # Calculate the accuracy score
        accuracy = accuracy_score(test_labels, predicted_labels)
        print(f"Accuracy: {accuracy}")
    except ValueError:
        print('failed to calculate accuracy score')
    try:
        # Calculate the F1 score
        f1 = f1_score(test_labels, predicted_labels, average='weighted')
        print(f"F1 Score: {f1}")
    except ValueError:
        print('failed to generate f1 score')


if __name__ == "__main__":
    train_intent_bart()
