import joblib
import pandas as pd

# Load Model
model = joblib.load("models/credit_scoring_model.pkl")

# Load Scaler
scaler = joblib.load("models/scaler.pkl")

# Load Feature Names
feature_names = joblib.load("models/feature_names.pkl")


def predict_credit(data):

    # Convert Dictionary to DataFrame
    df = pd.DataFrame([data])

    # Arrange columns correctly
    df = df[feature_names]

    # Scale Input
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)[0]

    probability = model.predict_proba(df_scaled)[0]

    return prediction, probability


if __name__ == "__main__":

    sample = {

        "Age": 30,
        "Annual_Income": 650000,
        "Loan_Amount": 200000,
        "Existing_Debts": 50000,
        "Credit_Utilization": 25,
        "Payment_History": 95,
        "Credit_History_Length": 8,
        "Savings_Balance": 250000,
        "EMI_Amount": 8000,
        "Debt_to_Income_Ratio": 15

    }

    prediction, probability = predict_credit(sample)

    print("=" * 50)

    if prediction == 1:
        print("Customer is Creditworthy")
    else:
        print("Customer is NOT Creditworthy")

    print()

    print("Probability")

    print(f"Not Creditworthy : {probability[0]*100:.2f}%")

    print(f"Creditworthy     : {probability[1]*100:.2f}%")

    print("=" * 50)