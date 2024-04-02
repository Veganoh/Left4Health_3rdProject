import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features, TFBertModel
from transformers import InputExample
import tensorflow as tf
import os
import pickle
import nltk
from tensorflow.keras.layers import Dense


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up
two_levels_up = os.path.abspath(os.path.join(current_dir, '..', '..'))
dataset_file_path = os.path.join(two_levels_up, 'dataset_with_intents.csv')
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

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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


def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder


# Assuming 'Intent' column for the intent labels
intents = df['Intent'].tolist()

# Encode disease and intent labels separately
disease_labels, disease_label_encoder = encode_labels(diseases)
intent_labels, intent_label_encoder = encode_labels(intents)


def create_tf_dataset(input_ids, attention_masks, disease_labels, intent_labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(({
                                                      'input_ids': input_ids,
                                                      'attention_mask': attention_masks
                                                  },
                                                  {'disease_output': disease_labels, 'intent_output': intent_labels}))
    dataset = dataset.shuffle(len(input_ids)).batch(batch_size)
    return dataset


def train_intent_and_disease_model():
    input_ids, attention_masks = encode_sentences(combined_texts)
    global disease_labels, intent_labels

    # Convert labels to tensorflow tensors
    disease_labels = tf.convert_to_tensor(disease_labels)
    intent_labels = tf.convert_to_tensor(intent_labels)

    # Create the dataset
    dataset = create_tf_dataset(input_ids, attention_masks, disease_labels, intent_labels)

    # Load the BERT model
    model = MultiTaskBert(len(disease_label_encoder.classes_),len(intent_label_encoder.classes_))

    # Prepare the model for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = {'disease_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'intent_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train the model
    model.fit(dataset, epochs=2)

    # Save the fine-tuned model and the label encoders for later use
    model.save('model/my_fine_tuned_bert_intents')
    with open('model/disease_label_encoder_intents.pkl', 'wb') as le_file:
        pickle.dump(disease_label_encoder, le_file)
    with open('model/intent_label_encoder_intents.pkl', 'wb') as le_file:
        pickle.dump(intent_label_encoder, le_file)


my_fine_tuned_bert = os.path.join(current_dir, 'model/my_fine_tuned_bert_intents')
# Load the fine-tuned model
model = tf.keras.models.load_model(my_fine_tuned_bert)


def predict_intent_bert_intents(input_sentence):
    # Load the fine-tuned BERT model
    # Load the trained model
    # Encode the input sentence
    input_ids, attention_mask = encode_sentences([input_sentence])
    global disease_labels
    # Make a prediction
    predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})

    # Convert logits to probabilities
    disease_probabilities = tf.nn.softmax(predictions['disease_output'], axis=-1).numpy()[0]

    # Get disease labels and intent label
    disease_labels = disease_label_encoder.classes_
    intent_label = intent_label_encoder.inverse_transform([tf.argmax(predictions['intent_output'], axis=1).numpy()[0]])[
        0]

    # Pair each disease label with its corresponding probability
    label_probabilities = list(zip(disease_labels, disease_probabilities))

    # Sort the pairs by probability in descending order
    label_probabilities.sort(key=lambda x: x[1], reverse=True)

    return {'label_probabilities': label_probabilities, 'intent':intent_label}


class MultiTaskBert(tf.keras.Model):
    def __init__(self, num_disease_labels, num_intent_labels):
        super(MultiTaskBert, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.disease_classifier = Dense(num_disease_labels, activation='softmax', name='disease_output')
        self.intent_classifier = Dense(num_intent_labels, activation='softmax', name='intent_output')

    def call(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Take the pooled output (CLS token)

        disease_logits = self.disease_classifier(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return {'disease_output': disease_logits, 'intent_output': intent_logits}


if __name__ == "__main__":
    train_intent_and_disease_model()
