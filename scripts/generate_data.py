"""Generate synthetic financial advisor data with realistic user names.

Based on the original generator used for virtual_financial_advisor_data.csv.
Changes from original:
  - 50 realistic user names instead of user_1..user_20
  - 50,000 rows instead of 5,200
  - Date range extended to 2023-01-01 through 2025-12-31
  - Output: data/virtual_financial_advisor_data_v2.csv
"""

import os
import pandas as pd
import numpy as np

np.random.seed(2026)

num_rows = 50000

user_ids = [
    "Aarav_Sharma", "Priya_Patel", "Rahul_Gupta", "Sneha_Reddy", "Vikram_Singh",
    "Ananya_Nair", "Rohan_Mehta", "Kavita_Joshi", "Arjun_Kumar", "Deepika_Rao",
    "Emily_Johnson", "James_Williams", "Olivia_Brown", "Ethan_Davis", "Sophia_Miller",
    "Carlos_Rodriguez", "Maria_Garcia", "Diego_Martinez", "Isabella_Lopez", "Alejandro_Hernandez",
    "Yuki_Tanaka", "Hiro_Nakamura", "Sakura_Yamamoto", "Kenji_Watanabe", "Aiko_Suzuki",
    "Fatima_Al_Hassan", "Omar_Khalil", "Layla_Ibrahim", "Hassan_Ahmed", "Amira_Mansour",
    "Lucas_Oliveira", "Ana_Santos", "Pedro_Costa", "Beatriz_Ferreira", "Rafael_Almeida",
    "Sarah_Mitchell", "David_Thompson", "Rachel_Anderson", "Michael_Taylor", "Jennifer_Wilson",
    "Wei_Zhang", "Li_Chen", "Mei_Wang", "Jun_Liu", "Xiao_Yang",
    "Nikolai_Petrov", "Elena_Volkov", "Ivan_Sokolov", "Olga_Kuznetsova", "Dmitri_Popov",
]

categories_income = ["Salary", "Bonus", "Interest"]
categories_expense = [
    "Groceries", "Rent", "Utilities", "Entertainment",
    "Dining", "Transport", "Healthcare", "Education", "Savings Transfer",
]

payment_methods = ["Credit Card", "Debit Card", "Cash", "Transfer"]

merchants = [
    "Amazon", "Walmart", "Netflix", "Uber", "Starbucks",
    "Local Grocery", "Electric Company", "Landlord", "Hospital", "School",
]

dates = pd.date_range("2023-01-01", "2025-12-31").to_pydatetime().tolist()

data = []
for _ in range(num_rows):
    user = np.random.choice(user_ids)
    date = np.random.choice(dates)
    is_income = np.random.choice([True, False], p=[0.15, 0.85])

    if is_income:
        category = np.random.choice(categories_income)
        amount = round(np.random.uniform(1000, 5000), 2)
    else:
        category = np.random.choice(categories_expense)
        amount = round(-np.random.uniform(5, 500), 2)

    payment_method = np.random.choice(payment_methods)
    merchant = np.random.choice(merchants)
    description = f"{category} payment at {merchant}"
    transaction_id = f"txn_{np.random.randint(1000000, 9999999)}"

    data.append([
        transaction_id, user, date, category, amount,
        payment_method, merchant, description,
    ])

df = pd.DataFrame(data, columns=[
    "transaction_id", "user_id", "date", "category", "amount",
    "payment_method", "merchant", "description",
])

out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
out_path = os.path.join(out_dir, "virtual_financial_advisor_data_v2.csv")
df.to_csv(out_path, index=False)
print(f"CSV file with {num_rows} rows created successfully at {out_path}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Categories: {sorted(df['category'].unique())}")
