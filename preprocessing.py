import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess_data(csv_path):
    """
    Load dataset and preprocess it.
    """

    # Load Dataset
    df = pd.read_csv(csv_path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill Missing Values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Features & Target
    X = df.drop("Creditworthy", axis=1)
    y = df["Creditworthy"]

    # Feature Names
    feature_names = list(X.columns)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale Features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save Scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Save Feature Names
    joblib.dump(feature_names, "models/feature_names.pkl")

    print("Preprocessing Completed Successfully")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    )


if __name__ == "__main__":

    preprocess_data("dataset/credit_scoring.csv")