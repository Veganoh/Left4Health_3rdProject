import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
distibert_tokenizer_path = os.path.join(current_dir, 'distil_bert_tokenizer.pkl')
distilbert_model_path = os.path.join(current_dir, 'distil_bert_model.pth')

# Load dataset
dataset_full = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../dataset_with_intents.csv')
df = pd.read_csv(dataset_full)

# Drop rows with missing values in any column
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Concatenate Disease and Intent columns with a hyphen separator
df['target'] = df['Disease'] + " - " + df['Intent']


# Assuming you have the dataframe `df` that was used with IntentDataset
unique_labels = df['target'].unique()
label_map = {label: idx for idx, label in enumerate(unique_labels)}

# Invert the label_map to create a map from indices to labels
index_to_label = {idx: label for label, idx in label_map.items()}

# Load tokenizer
with open(distibert_tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
try:
    # Load model
    loaded_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
        num_labels=len(df.target.unique()))

    loaded_model.load_state_dict(torch.load(distilbert_model_path))
    loaded_model.eval()
finally:
    print('Done loading')
# Function to predict
def predict_question_intent(question):
    # Tokenize input question
    inputs = tokenizer.encode_plus(
        question,
        None,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_token_type_ids=True,
        truncation=True
    )

    input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0)  # Add batch dimension
    attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0)  # Add batch dimension

    # Model inference
    with torch.no_grad():
        outputs = loaded_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Postprocess output
    predictions = torch.argmax(logits, dim=-1)
    predicted_index = predictions.cpu().numpy()[0]  # Move to CPU and convert to numpy
    # Map the predicted index to the corresponding label
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# Example usage
question = "What are the symptoms of psoriasis?"
predicted_intent = predict_question_intent(question)
print("Predicted Intent:", predicted_intent)