import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features
from transformers import InputExample
import tensorflow as tf
import os
import pickle

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
# Tokenize and encode sentences in BERT's format


def encode_sentences(sentences):
    global features, labels
    input_examples = [InputExample(guid=index, text_a=sentence, text_b=None, label=label) for index, (sentence, label) in enumerate(zip(sentences, labels))]
    features = glue_convert_examples_to_features(examples=input_examples, tokenizer=tokenizer, max_length=128, task='mnli', label_list=label_encoder.classes_)
    return features


features = encode_sentences(combined_texts)


# Convert to TensorFlow datasets
def convert_to_tf_dataset(feats, bs):
    # Convert our list of features into a TensorFlow dataset
    def gen():
        for ex in feats:
            yield (
                {
                   "input_ids": ex.input_ids,
                   "attention_mask": ex.attention_mask,
                   "token_type_ids": ex.token_type_ids,
                },
                ex.label
            )
    dataset = tf.data.Dataset.from_generator(
       gen,
       ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
       (
           {
               "input_ids": tf.TensorShape([None]),
               "attention_mask": tf.TensorShape([None]),
               "token_type_ids": tf.TensorShape([None]),
           },
           tf.TensorShape([]),
       ),
    )
    return dataset.shuffle(100).batch(bs)


def train_intent_bart():
    # Create TensorFlow datasets for training and validation
    batch_size = 32
    tf_dataset = convert_to_tf_dataset(features, batch_size)
    train_size = int(0.9 * len(tf_dataset))
    val_size = len(tf_dataset) - train_size
    train_dataset = tf_dataset.take(train_size)
    val_dataset = tf_dataset.skip(train_size)

    # Load the BERT model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

    # Fine-tune BERT on your dataset
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)

    # After fine-tuning, you can evaluate the model's performance on the validation set
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f'Validation loss: {val_loss}, validation accuracy: {val_accuracy}')

    # Save the fine-tuned model and the label encoder for later use
    label_encoder_path = os.path.join(current_dir, 'model/label_encoder.pkl')
    my_fine_tuned_bert = os.path.join(current_dir, 'model/my_fine_tuned_bert')
    model.save_pretrained(my_fine_tuned_bert)

    with open(label_encoder_path, 'wb') as le_file:
        pickle.dump(label_encoder, le_file)


def predict_intent(text_query):
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
