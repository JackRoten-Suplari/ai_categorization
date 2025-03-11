"""
BERT Categorization

This script uses BERT to categorize financial transactions based on their descriptions.

Not all Transactions come with descriptions
"""

# pip install torch transformers datasets scikit-learn pandas tqdm


# Step 1: Load & Preprocess Data
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load your dataset (CSV with 'Description' and 'Category' columns)
df = pd.read_csv("accounts_payable.csv")

# Ensure there are no missing values
df = df.dropna(subset=['Description', 'Category'])

# Convert categories to numerical labels
df['Category_Label'], category_mapping = pd.factorize(df['Category'])

# Split into training & testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Description'].tolist(), df['Category_Label'].tolist(), test_size=0.2, random_state=42
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the descriptions
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)


# Step 2: Create PyTorch Dataset
import torch

class TransactionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Create PyTorch datasets
train_dataset = TransactionDataset(train_encodings, train_labels)
test_dataset = TransactionDataset(test_encodings, test_labels)



from transformers import BertForSequenceClassification

# Load BERT with classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(category_mapping))



from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./bert_model",          # Output directory
    evaluation_strategy="epoch",        # Evaluate after each epoch
    save_strategy="epoch",              # Save after each epoch
    per_device_train_batch_size=8,      # Batch size
    per_device_eval_batch_size=8,
    num_train_epochs=4,                 # Adjust based on dataset size
    logging_dir="./logs",               # Logging
    logging_steps=100,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train Model
trainer.train()


# Step 5: Evaluate Model
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Evaluate
trainer.evaluate()

# Save model
model.save_pretrained("transaction_bert_model")
tokenizer.save_pretrained("transaction_bert_model")

# Load model later for inference
from transformers import BertForSequenceClassification


# Step 6: Save & Load Model
model = BertForSequenceClassification.from_pretrained("transaction_bert_model")
tokenizer = BertTokenizer.from_pretrained("transaction_bert_model")

# Step 7: Classify New Transactions
def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return category_mapping[prediction]

# Example Prediction
print(predict_category("Payment for software subscription"))
