import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features
from transformers import InputExample
import tensorflow as tf
import os
import pickle
import nltk


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up
two_levels_up = os.path.abspath(os.path.join(current_dir, '..', '..'))
dataset_file_path = os.path.join(two_levels_up, 'dataset_full.csv')
# Load the dataset
df = pd.read_csv(dataset_file_path)
# Preprocess the dataset
# Assume 'Question' column for the input and 'Disease' column for the intent labels
questions = df['Question'].tolist()
answers = df['Answer'].tolist()  # If you want to include answers in the model
diseases = df['Disease'].tolist()
# Combine questions and answers into a single text input
combined_texts = [question + " [SEP] " + answer for question, answer in zip(questions, answers)]
# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(diseases)
# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Create a TensorFlow dataset
def create_tf_dataset(input_ids, attention_masks, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(({
                                                      'input_ids': input_ids,
                                                      'attention_mask': attention_masks
                                                  }, labels))
    dataset = dataset.shuffle(len(input_ids)).batch(batch_size)
    return dataset


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
    model.fit(dataset, epochs=4)
    # Save the fine-tuned model and the label encoder for later use
    label_encoder_path = os.path.join(current_dir, 'model/label_encoder.pkl')
    my_fine_tuned_bert = os.path.join(current_dir, 'model/my_fine_tuned_bert')
    model.save_pretrained(my_fine_tuned_bert)

    with open(label_encoder_path, 'wb') as le_file:
        pickle.dump(label_encoder, le_file)


def predict_intent_bert(text_query):
    # Load the fine-tuned BERT model
    # Load the trained model
    my_fine_tuned_bert = os.path.join(current_dir, 'model/my_fine_tuned_bert')
    model = TFBertForSequenceClassification.from_pretrained(my_fine_tuned_bert)

    # Preprocess the text query
    encoded_dict = tokenizer.encode_plus(
        text_query,  # Input text
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attention masks.
        return_tensors='tf',  # Return TensorFlow tensors.
    )

    # Extract input IDs and attention masks from the encoded representation
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    # Make a prediction
    predictions = model(input_ids, attention_mask=attention_mask)

    # Convert logits to probabilities
    probabilities = tf.nn.softmax(predictions.logits, axis=-1).numpy()[0]

    probabilities = probabilities.astype(float)

    # Pair each label with its corresponding probability
    label_probabilities = list(zip(label_encoder.classes_, probabilities))

    # Sort the pairs by probability in descending order
    label_probabilities.sort(key=lambda x: x[1], reverse=True)

    return label_probabilities


if __name__ == "__main__":
    train_intent_bart()
