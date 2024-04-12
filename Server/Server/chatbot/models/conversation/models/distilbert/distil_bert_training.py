import os
import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report
from tqdm import tqdm

# Load dataset
dataset_full = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../dataset_with_intents.csv')
df = pd.read_csv(dataset_full)

# Drop rows with missing values in any column
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Concatenate Disease and Intent columns with a hyphen separator
df['target'] = df['Disease'] + " - " + df['Intent']

# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Save tokenizer
with open("distil_bert_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


# Define custom dataset class
class IntentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe  # Drop rows with missing data
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Map unique target labels to numerical indices
        self.label_map = {label: idx for idx, label in enumerate(self.data.target.unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):


        try:
            question = str(self.data.iloc[index].Question)
            target = str(self.data.iloc[index].target)

            inputs = self.tokenizer.encode_plus(
                question,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True
            )

            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            # Convert target label to tensor
            # Convert target label to numerical index
            target_index = self.label_map[target]
            target_tensor = torch.tensor(target_index, dtype=torch.long)
            return {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(mask, dtype=torch.long),
                'target': target_tensor
            }
        except KeyError:
            print(f"Index: {index}")


# Define model parameters
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 20

# Create train and test datasets
train_dataset = IntentDataset(train_df, tokenizer, MAX_LEN)
test_dataset = IntentDataset(test_df, tokenizer, MAX_LEN)

# Create data loaders
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)

test_sampler = SequentialSampler(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

# Define DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(df.target.unique())
)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['target']

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=target)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")


# Save model
torch.save(model.state_dict(), "distil_bert_model.pth")


# Evaluate the model
model.eval()

# Initialize lists to store true and predicted labels
true_labels = []
pred_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predictions = torch.max(logits, dim=1)

        # Move preds and labels to CPU
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # Store predictions and true labels
        true_labels.extend(labels)
        pred_labels.extend(predictions)

# Calculate accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate F1 score
f1 = f1_score(true_labels, pred_labels, average='weighted')
print(f"F1 Score: {f1:.2f}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report for precision, recall, f1-score
print("Classification Report:")
print(classification_report(true_labels, pred_labels))

# Calculate the average training loss
avg_train_loss = total_loss / len(train_loader)
print(f"Average training loss: {avg_train_loss:.2f}")

