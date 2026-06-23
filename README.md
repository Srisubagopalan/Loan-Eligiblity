# Credit Scoring Model

## Project Description

This project predicts whether a customer is creditworthy using Machine Learning.

---

## Features

- Credit Risk Prediction
- Logistic Regression
- Decision Tree
- Random Forest
- Data Preprocessing
- Model Evaluation
- Streamlit Web Application

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

## Project Structure

```
Credit_Scoring_Project/
│
├── dataset/
│   └── credit_scoring.csv
│
├── models/
│   ├── credit_scoring_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── generate_dataset.py
├── preprocessing.py
├── train_model.py
├── prediction.py
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1

Generate Dataset

```bash
python generate_dataset.py
```

### Step 2

Preprocess Data

```bash
python preprocessing.py
```

### Step 3

Train Model

```bash
python train_model.py
```

### Step 4

Prediction

```bash
python prediction.py
```

### Step 5

Run Streamlit

```bash
python -m streamlit run app.py
```

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

## Author

Credit Scoring Machine Learning Project