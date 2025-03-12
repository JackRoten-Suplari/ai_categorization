# AI Spend Categorization

This project uses Anthropic's Claude AI to automatically categorize spend accounts into direct, indirect, and unaddressable categories.

## Features

- Extracts unique account names from transaction data
- Uses Anthropic's Claude AI to categorize accounts
- Processes accounts in batches for efficiency
- Includes retry logic and error handling
- Saves results to CSV for easy analysis

## Prerequisites

- Python 3.8 or higher
- Poetry (Python package manager)
- Anthropic API key

For testing data locally download data from google drive in CS/_test_data and place in data/ dir

## Installation

1. Clone the repository:
```bash
git clone https://github.com/suplari/ai_categorization.git
cd ai_categorization
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up your Anthropic API key:
Create a `.env` file in the project root:
```bash
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```
- Replace `your-api-key-here` with your actual Anthropic API key

## Usage

To test account categorization method:
1. Activate the Poetry virtual environment:
```bash
poetry shell
```

2. Run the categorization script:
```bash
poetry run categorize
```

Or alternatively:
```bash
python src/anthropic_utils_dev.py
```

The script will:
1. Load transaction data from `data/coopervision_Suplari_Transactions_sample.csv`
2. Extract unique account names
3. Use Claude to categorize each account
4. Save results to `account_categories.csv`

## Input Data Format

The input CSV file should contain at least an "account name" column. Place your data file at:
```
data/coopervision_Suplari_Transactions_sample.csv
```

## Output Format

The script generates `account_categories.csv` with two columns:
- `account_name`: The original account name
- `spend_category`: The AI-assigned category (direct/indirect/unaddressable)

## Development

To contribute to the project:

1. Install development dependencies:
```bash
poetry install --with dev
```

2. Format code:
```bash
poetry run black src/
poetry run isort src/
```

## Security Notes

- Never commit your `.env` file or expose your API key
- The `.env` file is included in `.gitignore`
- Always use environment variables for sensitive credentials

