import openai
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# OpenAI API Key
openai.api_key = "your_openai_api_key"

def categorize_with_ai(description):
    """
    Uses OpenAI's API to categorize transactions based on descriptions.
    """
    prompt = f"Categorize the following financial transaction: '{description}'. Categories: [Invoice Payment, Refund, Payroll Expense, Travel Expense, Software/Subscription, Office Expense, Reversal/Correction, Miscellaneous]. Return only the category."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "Miscellaneous"

# Convert to UDF for PySpark
categorize_udf = udf(categorize_with_ai, StringType())

# Apply AI Categorization
df = df.withColumn("AI_Category", categorize_udf(df["Description"]))
df.show(10, truncate=False)
