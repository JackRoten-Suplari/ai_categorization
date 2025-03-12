import torch
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from TransactionDataset import TransactionDataset
from HierarchicalBertClassifier import HierarchicalBertClassifier
from bert_categorization_utils import (
    load_and_preprocess_data,
    prepare_hierarchical_labels,
    train_model,
    predict_batch,
    save_predictions_to_csv
)

def main():
    """
    Usage of BERT Model for Categorization of Spend
    """
    # Define fields to use
    fields_to_use = [
        'normalized vendor',
        'account name',
    ]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    print(f"Using fields: {', '.join(fields_to_use)}")
    transactions_df, category_mappings, label_encoders = load_and_preprocess_data(fields_to_use)
    
    # Print some data statistics
    print(f"\nDataset statistics:")
    print(f"Number of transactions: {len(transactions_df)}")
    print("Number of categories per level:")
    for level, mapping in enumerate(category_mappings, 1):
        print(f"Level {level}: {len(mapping)} categories")
    
    # Prepare labels
    print("\nPreparing labels...")
    hierarchical_labels = prepare_hierarchical_labels(transactions_df, category_mappings)
    
    # Split data
    print("\nSplitting data into train and validation sets...")
    train_idx, val_idx = train_test_split(
        range(len(transactions_df)),
        test_size=0.2,
        random_state=42,
        stratify=[l if l != -1 else 0 for l in hierarchical_labels[0]]  # Use first level for stratification
    )
    
    train_labels = [[labels[i] for i in train_idx] for labels in hierarchical_labels]
    val_labels = [[labels[i] for i in val_idx] for labels in hierarchical_labels]
    
    print(f"Training set size: {len(train_idx)}")
    print(f"Validation set size: {len(val_idx)}")
    
    # Initialize tokenizer and create datasets
    print("\nInitializing BERT tokenizer and creating datasets...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TransactionDataset(transactions_df, train_idx, train_labels, tokenizer, fields_to_use)
    val_dataset = TransactionDataset(transactions_df, val_idx, val_labels, tokenizer, fields_to_use)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize and train model
    print("\nInitializing model...")
    num_categories_per_level = [len(mapping) for mapping in category_mappings]
    model = HierarchicalBertClassifier(num_categories_per_level).to(device)
    
    print("\nTraining model...")
    train_model(model, train_dataloader, val_dataloader, device, category_mappings)
    
    # Save the model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'category_mappings': category_mappings,
        'label_encoders': label_encoders,
        'fields_to_use': fields_to_use
    }, 'coopervision_bert_model.pt')
    
    # Generate predictions for all transactions
    print("\nGenerating predictions for all transactions...")
    all_predictions = predict_batch(model, tokenizer, transactions_df, fields_to_use, label_encoders, device)
    
    # Save predictions to CSV
    save_predictions_to_csv(transactions_df, all_predictions, 'coopervision_predictions.csv')
    
    # Show an example prediction
    print("\nExample Prediction:")
    print("Input fields:")
    for field in fields_to_use:
        if field in transactions_df.columns:
            value = transactions_df[field].iloc[0]
            print(f"{field}: {value}")
    print("\nPredicted categories:")
    for level, predictions in enumerate(all_predictions, 1):
        print(f"Level {level}: {predictions[0]}")

if __name__ == "__main__":
    main() 