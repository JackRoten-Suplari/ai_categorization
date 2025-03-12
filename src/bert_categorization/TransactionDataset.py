
import pandas as pd
import torch
from torch.utils.data import Dataset


class TransactionDataset(Dataset):
    def __init__(self, transactions_df, indices, labels, tokenizer, fields_to_use=None, max_length=256):
        if fields_to_use is None:
            fields_to_use = ['description', 'normalized vendor', 'account_name', 'cost center name']
        
        # Combine multiple fields into a single text
        texts = []
        for idx in indices:
            text_parts = []
            for field in fields_to_use:
                if field in transactions_df.columns:
                    value = str(transactions_df[field].iloc[idx])
                    if pd.notna(value) and value != 'nan':
                        text_parts.append(f"{field}: {value}")
            texts.append(" | ".join(text_parts))
        
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