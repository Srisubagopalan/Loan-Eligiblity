import pandas as pd
import numpy as np
import os

# Random seed for reproducibility
np.random.seed(42)

# Number of records
n = 5000

# Create dataset
data = {
    "Age": np.random.randint(21, 60, n),
    "Annual_Income": np.random.randint(200000, 1500000, n),
    "Loan_Amount": np.random.randint(50000, 1000000, n),
    "Existing_Debts": np.random.randint(0, 500000, n),
    "Credit_Utilization": np.random.randint(5, 100, n),
    "Payment_History": np.random.randint(50, 100, n),
    "Credit_History_Length": np.random.randint(1, 30, n),
    "Savings_Balance": np.random.randint(10000, 1000000, n),
    "EMI_Amount": np.random.randint(1000, 50000, n)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature Engineering
df["Debt_to_Income_Ratio"] = (
    df["Existing_Debts"] / df["Annual_Income"] * 100
).round(2)

# Target Variable
df["Creditworthy"] = np.where(
    (df["Payment_History"] >= 75) &
    (df["Credit_Utilization"] <= 40) &
    (df["Debt_to_Income_Ratio"] <= 40),
    1,
    0
)

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Save CSV
df.to_csv("dataset/credit_scoring.csv", index=False)

print("=" * 50)
print("Dataset Generated Successfully")
print("=" * 50)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print("\nSaved at : dataset/credit_scoring.csv")
print(df.head())