"""
CooperVision BERT Categorization

This script uses BERT to categorize financial transactions into hierarchical categories (levels 1-5)
based on their descriptions.

TODO: Add more fields to this classification
"""

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


def load_and_preprocess_data(fields_to_use=None):
    if fields_to_use is None:
        fields_to_use = [
            'description',
            'normalized vendor',
            'account_name',
            'cost center name',
            'amount',
            'payment method'
        ]
    
    # Load categories
    categories_df = pd.read_csv("data/suplari_coopervision_Categories.csv")
    
    # Load transactions with low_memory=False to handle mixed types
    transactions_df = pd.read_csv("data/coopervision_Suplari_Transactions_sample.csv", low_memory=False)
    
    # Convert all specified fields to string and clean them
    for field in fields_to_use:
        if field in transactions_df.columns:
            transactions_df[field] = transactions_df[field].astype(str)
            # Clean the field values
            transactions_df[field] = transactions_df[field].apply(lambda x: x.strip() if isinstance(x, str) else x)
            transactions_df[field] = transactions_df[field].replace('nan', '')
    
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

def predict_batch(model, tokenizer, transactions_df, fields_to_use, label_encoders, device, batch_size=32):
    model.eval()
    predictions = [[] for _ in range(len(label_encoders))]
    
    # Create batches of combined text
    all_texts = []
    for _, row in transactions_df.iterrows():
        text_parts = []
        for field in fields_to_use:
            if field in transactions_df.columns:
                value = str(row[field])
                if pd.notna(value) and value != 'nan':
                    text_parts.append(f"{field}: {value}")
        all_texts.append(" | ".join(text_parts))
    
    # Process in batches
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            
            # Get predictions for each level
            for level, (level_logits, encoder) in enumerate(zip(logits, label_encoders)):
                if level_logits.size(1) == 0:  # Skip empty categories
                    predictions[level].extend([''] * len(batch_texts))
                    continue
                    
                # Handle both single and batch predictions
                if len(batch_texts) == 1:
                    pred_idx = level_logits.squeeze(0).argmax().item()
                    predictions[level].append(encoder[pred_idx])
                else:
                    pred_indices = level_logits.argmax(dim=1)
                    pred_categories = [encoder[idx.item()] for idx in pred_indices]
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

