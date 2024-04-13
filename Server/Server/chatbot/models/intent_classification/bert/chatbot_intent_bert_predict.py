import os
import pickle

from sklearn.preprocessing import LabelEncoder
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = None
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
label_encoder_path = os.path.join(current_dir, 'model/label_encoder.pkl')
try:
    my_fine_tuned_bert = os.path.join(current_dir, 'model/my_fine_tuned_bert')
    model = TFBertForSequenceClassification.from_pretrained(my_fine_tuned_bert)
    # Load the label encoder

    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
except OSError:
    print('Model does not exist, must be loaded')
finally:
    print('Loading of model completed')


def predict_intent_bert(text_query):
    # Load the fine-tuned BERT model
    # Load the trained model
    if model is None:
        return None
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
    print(label_probabilities)
    return label_probabilities[0]