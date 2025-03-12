import pandas as pd
from typing import List, Dict

import os
from tqdm import tqdm
import time

def extract_unique_account_names(df: pd.DataFrame) -> List[str]:
    """Extract unique values from the 'account name' field in a DataFrame."""
    if 'account name' not in df.columns:
        raise KeyError("DataFrame must contain an 'account name' column")
    
    # Extract unique values and sort them
    unique_accounts = sorted(df['account name'].unique().tolist())
    
    return unique_accounts

def extract_keywords_from_filename(df: pd.DataFrame) -> pd.DataFrame:
    if 'file_name' not in df.columns:
        raise KeyError("DataFrame must contain a 'file_name' column")
    
    # Create case-insensitive pattern matches for both keywords
    mask_indirect = df['file_name'].str.contains('indirect', case=False, na=False)
    mask_paid = df['file_name'].str.contains('Paid', case=True, na=False)
    
    # Combine masks with OR operation
    combined_mask = mask_indirect | mask_paid
    
    # Filter DataFrame
    filtered_df = df[combined_mask].copy()
    
    # Add columns indicating which keyword was matched
    filtered_df['contains_indirect'] = mask_indirect[combined_mask]
    filtered_df['contains_paid'] = mask_paid[combined_mask]
    
    return filtered_df

# TODO: file_name tells us what is indirect, o

