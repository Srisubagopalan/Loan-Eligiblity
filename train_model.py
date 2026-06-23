import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import preprocess_data


# Load and preprocess data
X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
    "dataset/credit_scoring.csv"
)


# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

best_model = None
best_accuracy = 0

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report")
    print(classification_report(y_test, prediction))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model


# Save Best Model
os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/credit_scoring_model.pkl"
)

print("\n")
print("=" * 60)
print("BEST MODEL SAVED SUCCESSFULLY")
print("=" * 60)
print(f"Best Accuracy : {best_accuracy:.4f}")
print("Saved File : models/credit_scoring_model.pkl")