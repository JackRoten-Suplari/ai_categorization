# ai_categorization
Testing grounds for AI and ML categorization.

1. First install the required dependencies:
```bash
pip install -r requirements.txt

```

Suplari's goals to categorize transactions and complete the spend cube includes a few cases from simpleset to more complex:

1. Categorizing transactions based on a set of given rules and hierachy
2.  Categorizing transactions based on features such as description, account, cost center, business unit and hierachy
3.  Categorizing transactions based on features such as description, account, cost center, business unit no provided hierachy

Categorization based on features requires as many dimensions as possible. It is yet to be determined what the most optimal number of features is and how best they can be used. Below is an extensive list of potential feaures, of which, we can look to a LLM to aid in understanding our customer's data.



Features of our customers data that can aid in categorizating transactions:

1. Transaction Metadata (General Identifiers)
Transaction ID – Unique identifier for each transaction.
Timestamp – Date and time of the transaction (used for trend analysis).
Location – Geographical coordinates or address where the transaction occurred.
Merchant Name / Category – Type of business where the transaction happened.
Channel Used – Online, mobile, POS (Point of Sale), ATM, etc.

2. Financial Features
Transaction Amount – Amount spent in the transaction.
Transaction Type – Credit, debit, refund, payment, withdrawal.
Currency Type – USD, EUR, GBP, etc.
Exchange Rate (if applicable) – Useful in cross-border transactions.
Payment Method – Credit card, debit card, PayPal, wire transfer, cryptocurrency, etc.
Recurring vs. One-Time – Subscription payments vs. one-time purchases.
Balance Before & After – Useful for tracking changes in financial accounts.

3. Customer & Behavioral Features
Customer ID – Unique identifier for the customer.
Account Type – Individual, business, premium, basic.
Customer Segmentation – High-value customers, regular users, occasional users.
Historical Transaction Frequency – Daily, weekly, monthly transactions.
Average Transaction Value – Average amount spent per transaction.
Transaction Patterns – Predictable spending habits vs. anomalies.
Time of Day & Day of Week – Morning, evening, weekend vs. weekday transactions.

4. Risk & Security Indicators
IP Address & Geolocation – Helps detect location-based fraud.
Device Fingerprint – Identifies unique device activity.
Previous Fraud Flag – Whether the customer or transaction was previously marked as fraudulent.
Velocity Checks – Too many transactions in a short period.
Unusual Amount – Higher than normal spending.
Cross-Border Transaction – Domestic vs. international.
MCC Code (Merchant Category Code) – Identifies high-risk merchants (e.g., gambling, cryptocurrency exchanges).

5. Industry-Specific Features
E-commerce – Cart size, items purchased, discount applied.
Banking – Loan repayment, overdraft usage, account deposits.
Healthcare – Type of medical service, insurance coverage.
Retail – Loyalty points earned, purchase category.
B2B Transactions – Invoice type, payment terms.

6. Text-Based Features (NLP on Transaction Descriptions)
Transaction Description Keywords – Extract key words from transaction narratives.
Sentiment Analysis – Used in disputes, chargebacks, or feedback.
Categorization via NLP – Assigning transactions to predefined categories using natural language processing.

Feature Engineering for Categorization
Clustering Techniques – Group transactions into clusters based on similarities (e.g., K-Means, DBSCAN).
Anomaly Detection – Isolation forests or autoencoders for identifying outliers.
Time Series Features – Seasonality, trend, cyclic patterns.
Graph-Based Features – Network of transactions between accounts.

Which Features to Use?
For Fraud Detection → Use security indicators, velocity checks, location & device data.
For Customer Segmentation → Use behavioral features, transaction patterns, and spending habits.
For Financial Risk Analysis → Use transaction amount, balance, and recurring nature.
