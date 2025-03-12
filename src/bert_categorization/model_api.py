# pip install fastapi uvicorn torch transformers

# tep 1: Load the Trained Model in FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize FastAPI app
app = FastAPI()

# Load the fine-tuned BERT model & tokenizer
MODEL_PATH = "transaction_bert_model"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Define the transaction categories (same mapping as during training)
category_mapping = ["Invoice Payment", "Refund", "Payroll Expense", "Travel Expense", 
                    "Software/Subscription", "Office Expense", "Reversal/Correction", "Miscellaneous"]

# Define request payload format
class TransactionRequest(BaseModel):
    description: str

# Function to predict transaction category
def predict_category(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return category_mapping[prediction]

# Define API endpoint for classification
@app.post("/predict")
def classify_transaction(request: TransactionRequest):
    category = predict_category(request.description)
    return {"description": request.description, "predicted_category": category}

# Step 2: Run the API Server
# uvicorn transaction_api:app --host 0.0.0.0 --port 8000 --reload

# Step 3: Test the API
"""
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"description": "Payment for cloud software subscription"}'

Expected Output:
{
  "description": "Payment for cloud software subscription",
  "predicted_category": "Software/Subscription"
}

"""
