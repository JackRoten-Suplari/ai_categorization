"""
CooperVision BERT Categorization

This script uses BERT to categorize financial transactions into hierarchical categories (levels 1-5)
based on their descriptions.

TODO: Add more fields to this classification
"""

import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

class HierarchicalBertClassifier(nn.Module):
    def __init__(self, num_categories_per_level):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        
        # Create separate classifiers for each level
        self.classifiers = nn.ModuleList([
            nn.Linear(768, num_cats) for num_cats in num_categories_per_level
        ])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each level
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits

class TransactionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    
    def __len__(self):
        return len(self.labels[0])  # Use first level's labels for length
    
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item['labels'] = [torch.tensor(level_labels[idx]) for level_labels in self.labels]
        return item

def load_and_preprocess_data():
    # Load categories
    categories_df = pd.read_csv("data/suplari_coopervision_Categories.csv")
    
    # Load transactions with low_memory=False to handle mixed types
    transactions_df = pd.read_csv("data/coopervision_Suplari_Transactions_sample.csv", low_memory=False)
    transactions_df['description'] = transactions_df['description'].astype(str)
    # Create category mappings for each level
    category_mappings = []
    label_encoders = []
    
    for level in range(1, 6):  # Levels 1-5
        level_col = f'level {level}'
        if level_col in categories_df.columns:
            unique_categories = categories_df[level_col].dropna().unique()
            mapping = {cat: idx for idx, cat in enumerate(unique_categories)}
            category_mappings.append(mapping)
            label_encoders.append({idx: cat for idx, cat in enumerate(unique_categories)})
    
    return transactions_df, category_mappings, label_encoders

def prepare_hierarchical_labels(transactions_df, category_mappings):
    # Initialize labels for each level
    labels = [[] for _ in range(len(category_mappings))]
    
    for idx, row in transactions_df.iterrows():
        for level, mapping in enumerate(category_mappings):
            category = row.get(f'level {level+1}')
            if pd.isna(category) or category not in mapping:
                labels[level].append(-1)  # Use -1 for missing categories
            else:
                labels[level].append(mapping[category])
    
    return labels

def train_model(model, train_dataloader, val_dataloader, device, category_mappings, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore -1 labels
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = [level_labels.to(device) for level_labels in batch['labels']]
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            # Calculate loss for each level and sum
            batch_losses = []
            for level_logits, level_labels in zip(logits, labels):
                if torch.any(level_labels != -1):
                    level_loss = criterion(level_logits, level_labels)
                    batch_losses.append(level_loss)
            
            if batch_losses:  # Only process if we have valid losses
                loss = torch.stack(batch_losses).sum()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        correct_predictions = [0] * len(category_mappings)
        total_predictions = [0] * len(category_mappings)
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = [level_labels.to(device) for level_labels in batch['labels']]
                
                logits = model(input_ids, attention_mask)
                
                # Calculate validation loss
                batch_losses = []
                for level_logits, level_labels in zip(logits, labels):
                    if torch.any(level_labels != -1):
                        level_loss = criterion(level_logits, level_labels)
                        batch_losses.append(level_loss)
                
                if batch_losses:
                    val_loss += torch.stack(batch_losses).sum().item()
                    val_batches += 1
                
                # Calculate accuracy
                for level, (level_logits, level_labels) in enumerate(zip(logits, labels)):
                    mask = level_labels != -1
                    if torch.any(mask):
                        predictions = level_logits[mask].argmax(dim=1)
                        correct_predictions[level] += (predictions == level_labels[mask]).sum().item()
                        total_predictions[level] += mask.sum().item()
        
        # Print metrics
        print(f'\nEpoch {epoch+1}:')
        if num_batches > 0:
            print(f'Training Loss: {total_loss/num_batches:.4f}')
        if val_batches > 0:
            print(f'Validation Loss: {val_loss/val_batches:.4f}')
        for level in range(len(category_mappings)):
            if total_predictions[level] > 0:
                accuracy = correct_predictions[level] / total_predictions[level]
                print(f'Level {level+1} Accuracy: {accuracy:.4f}')

def predict_categories(model, tokenizer, text, label_encoders, device):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    predictions = []
    for level, (level_logits, encoder) in enumerate(zip(logits, label_encoders)):
        pred_idx = level_logits.argmax(dim=1).item()
        pred_category = encoder[pred_idx]
        predictions.append(pred_category)
    
    return predictions

def predict_batch(model, tokenizer, texts, label_encoders, device, batch_size=32):
    model.eval()
    predictions = [[] for _ in range(len(label_encoders))]
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            
            # Get predictions for each level
            for level, (level_logits, encoder) in enumerate(zip(logits, label_encoders)):
                pred_idx = level_logits.argmax(dim=1)
                pred_categories = [encoder[idx.item()] for idx in pred_idx]
                predictions[level].extend(pred_categories)
    
    return predictions

def save_predictions_to_csv(transactions_df, predictions, output_file='predictions.csv'):
    # Create a copy of the original DataFrame
    output_df = transactions_df.copy()
    
    # Add predicted categories
    for level in range(len(predictions)):
        output_df[f'predicted_level_{level+1}'] = predictions[level]
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    transactions_df, category_mappings, label_encoders = load_and_preprocess_data()
    
    # Prepare labels
    hierarchical_labels = prepare_hierarchical_labels(transactions_df, category_mappings)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = [], [], [], []
    for level_labels in hierarchical_labels:
        train_idx, val_idx = train_test_split(
            range(len(level_labels)),
            test_size=0.2,
            random_state=42,
            stratify=[l if l != -1 else 0 for l in level_labels]
        )
        train_labels.append([level_labels[i] for i in train_idx])
        val_labels.append([level_labels[i] for i in val_idx])
    
    # Use lowercase 'description' column name
    train_texts = [transactions_df['description'].iloc[i] for i in train_idx]
    val_texts = [transactions_df['description'].iloc[i] for i in val_idx]
    
    # Initialize tokenizer and create datasets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TransactionDataset(train_texts, train_labels, tokenizer)
    val_dataset = TransactionDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize and train model
    num_categories_per_level = [len(mapping) for mapping in category_mappings]
    model = HierarchicalBertClassifier(num_categories_per_level).to(device)
    
    train_model(model, train_dataloader, val_dataloader, device, category_mappings)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'category_mappings': category_mappings,
        'label_encoders': label_encoders
    }, 'coopervision_bert_model.pt')
    
    # Generate predictions for all transactions
    print("\nGenerating predictions for all transactions...")
    all_descriptions = transactions_df['description'].tolist()
    all_predictions = predict_batch(model, tokenizer, all_descriptions, label_encoders, device)
    
    # Save predictions to CSV
    save_predictions_to_csv(transactions_df, all_predictions, 'coopervision_predictions.csv')
    
    # Show an example prediction
    example_idx = 0  # Show first transaction as example
    print("\nExample Prediction:")
    print(f"Text: {all_descriptions[example_idx]}")
    for level, predictions in enumerate(all_predictions, 1):
        print(f"Level {level}: {predictions[example_idx]}")

if __name__ == "__main__":
    main() 