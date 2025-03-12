import pandas as pd
from typing import List, Dict
import os
import time
from tqdm import tqdm
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_anthropic_api_key() -> str:
    """
    Get the Anthropic API key from environment variables.
    
    Returns:
        str: The API key if found
        
    Raises:
        ValueError: If API key is not found in environment variables
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables. "
            "Please set it using one of these methods:\n"
            "1. Add to ~/.config/fish/config.fish: set -gx ANTHROPIC_API_KEY 'your-key'\n"
            "2. Set in current session: set -x ANTHROPIC_API_KEY 'your-key'\n"
            "3. Add to .env file in project root: ANTHROPIC_API_KEY=your-key"
        )
    return api_key

def extract_unique_account_names(df: pd.DataFrame) -> List[str]:
    """Extract unique values from the 'account name' field in a DataFrame."""
    if 'account name' not in df.columns:
        raise KeyError("DataFrame must contain an 'account name' column")
    
    # Extract unique values and sort them
    unique_accounts = sorted(df['account name'].unique().tolist())
    
    return unique_accounts

def categorize_accounts_with_anthropic(
    accounts: List[str], 
    batch_size: int = 10,
    api_key: str = None
) -> Dict[str, str]:
    """
    Categorize account names as 'direct', 'indirect', or 'unaddressable' using Anthropic's API.
    
    Args:
        accounts (List[str]): List of account names to categorize
        batch_size (int): Number of accounts to process in each API call
        api_key (str): Anthropic API key. If None, will look for ANTHROPIC_API_KEY env variable
        
    Returns:
        Dict[str, str]: Dictionary mapping account names to their categories
    """
    # Initialize Anthropic client
    api_key = api_key or get_anthropic_api_key()
    client = Anthropic(api_key=api_key)
    
    # Process accounts in batches
    results = {}
    for i in tqdm(range(0, len(accounts), batch_size)):
        batch = accounts[i:i + batch_size]
        
        # Construct the prompt
        prompt = """Please categorize each of the following account names as either "direct", "indirect", or "unaddressable" spend.

                    Definitions:
                    - Direct spend: Directly related to the production of goods/services (e.g., raw materials, production equipment)
                    - Indirect spend: Supporting business operations (e.g., office supplies, consulting services)
                    - Unaddressable spend: Cannot be influenced or negotiated (e.g., taxes, regulatory fees)

                    Account names to categorize:
                    {}

                    Please respond in a JSON format like this:
                    {{"Account Name 1": "direct", "Account Name 2": "indirect", ...}}

                    Only include the JSON response, no other text.""".format("\n".join(batch))

        # Make API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Parse the response
                try:
                    import json
                    batch_results = json.loads(response.content[0].text)
                    results.update(batch_results)
                    break
                except json.JSONDecodeError:
                    if attempt == max_retries - 1:
                        print(f"Failed to parse response for batch starting with {batch[0]}")
                        continue
            
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to process batch starting with {batch[0]}: {str(e)}")
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # Rate limiting
        time.sleep(1)
    
    return results

def categorize_df_accounts(
    df: pd.DataFrame,
    batch_size: int = 10,
    api_key: str = None
) -> pd.DataFrame:
    """
    Add spend category classifications to a DataFrame based on account names.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing 'account name' column
        batch_size (int): Number of accounts to process in each API call
        api_key (str): Anthropic API key
        
    Returns:
        pd.DataFrame: Original DataFrame with new 'spend_category' column
    """
    # Get unique accounts and their categories
    unique_accounts = extract_unique_account_names(df)
    categories = categorize_accounts_with_anthropic(unique_accounts, batch_size, api_key)
    
    # Add categories to DataFrame
    df_with_categories = df.copy()
    df_with_categories['spend_category'] = df_with_categories['account name'].map(categories)
    
    return df_with_categories

def save_account_categories(categories: Dict[str, str], output_file: str = 'account_categories.csv'):
    """
    Save account categories to a CSV file.
    
    Args:
        categories (Dict[str, str]): Dictionary mapping account names to categories
        output_file (str): Path to save the CSV file
    """
    df = pd.DataFrame(
        [(account, category) for account, category in categories.items()],
        columns=['account_name', 'spend_category']
    )
    df.to_csv(output_file, index=False)
    print(f"Categories saved to {output_file}")

def main():
    try:
        print("Loading data...")
        df = pd.read_csv("data/coopervision_Suplari_Transactions_sample.csv")
        print(f"Loaded {len(df)} transactions")
        
        print("\nExtracting unique account names...")
        unique_accounts = extract_unique_account_names(df)
        print(f"Found {len(unique_accounts)} unique account names")
        
        print("\nCategorizing accounts...")
        categories = categorize_accounts_with_anthropic(unique_accounts)
        
        print("\nSaving results...")
        save_account_categories(categories)
        
        print("\nDone! Check account_categories.csv for results.")
        
    except FileNotFoundError:
        print("Error: Could not find the data file. Please make sure 'data/coopervision_Suplari_Transactions_sample.csv' exists.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
