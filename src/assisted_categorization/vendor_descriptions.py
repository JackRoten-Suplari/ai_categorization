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

def extract_unique_vendors(df: pd.DataFrame, vendor_column: str = 'normalized vendor') -> List[str]:
    """
    Extract unique vendor names from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing vendor information
        vendor_column (str): Name of the column containing vendor names
        
    Returns:
        List[str]: List of unique vendor names sorted alphabetically
    """
    if vendor_column not in df.columns:
        raise KeyError(f"DataFrame must contain a '{vendor_column}' column")
    
    # Extract unique values and sort them
    unique_vendors = sorted(df[vendor_column].unique().tolist())
    
    return unique_vendors

def get_vendor_descriptions(
    vendors: List[str],
    batch_size: int = 5,
    api_key: str = None
) -> Dict[str, str]:
    """
    Get descriptions for a list of vendors using Anthropic's Claude.
    
    Args:
        vendors (List[str]): List of vendor names to get descriptions for
        batch_size (int): Number of vendors to process in each API call
        api_key (str): Anthropic API key. If None, will look for ANTHROPIC_API_KEY env variable
        
    Returns:
        Dict[str, str]: Dictionary mapping vendor names to their descriptions
    """
    # Initialize Anthropic client
    api_key = api_key or get_anthropic_api_key()
    client = Anthropic(api_key=api_key)
    
    # Process vendors in batches
    results = {}
    for i in tqdm(range(0, len(vendors), batch_size)):
        batch = vendors[i:i + batch_size]
        
        # Construct the prompt
        prompt = """For each of the following vendors, describe what products or services they provide. 
        Keep each description concise (1-2 sentences).
        
        Vendors to describe:
        {}
        
        Please respond in a JSON format like this:
        {{"Vendor Name 1": "Description of vendor 1's products/services", 
          "Vendor Name 2": "Description of vendor 2's products/services"}}
        
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

def save_vendor_descriptions(descriptions: Dict[str, str], output_file: str = 'vendor_descriptions.csv'):
    """
    Save vendor descriptions to a CSV file.
    
    Args:
        descriptions (Dict[str, str]): Dictionary mapping vendor names to descriptions
        output_file (str): Path to save the CSV file
    """
    df = pd.DataFrame(
        [(vendor, desc) for vendor, desc in descriptions.items()],
        columns=['vendor_name', 'description']
    )
    df.to_csv(output_file, index=False)
    print(f"Descriptions saved to {output_file}")

def main():
    try:
        print("Loading data...")
        df = pd.read_csv("data/coopervision_Suplari_Transactions_sample.csv")
        print(f"Loaded {len(df)} transactions")
        
        print("\nExtracting unique vendors...")
        unique_vendors = extract_unique_vendors(df)
        print(f"Found {len(unique_vendors)} unique vendors")
        
        print("\nGetting vendor descriptions...")
        descriptions = get_vendor_descriptions(unique_vendors)
        
        print("\nSaving results...")
        save_vendor_descriptions(descriptions)
        
        print("\nDone! Check vendor_descriptions.csv for results.")
        
    except FileNotFoundError:
        print("Error: Could not find the data file. Please make sure 'data/coopervision_Suplari_Transactions_sample.csv' exists.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 