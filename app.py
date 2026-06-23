import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent
model = joblib.load(BASE_DIR / "models" / "credit_scoring_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
feature_names = joblib.load(BASE_DIR / "models" / "feature_names.pkl")

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="centered"
)
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #2C1B11, #FFFFFF);
}

/* Title */
.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:white;
}

/* Subtitle */
.subtitle{
    text-align:center;
    color:#CBD5E1;
    font-size:18px;
}

/* Card */
.card{
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    padding:25px;
    border-radius:18px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.3);
    margin-bottom:20px;
    color:white;
}

/* Button */
.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    background:#38BDF8;
    color:white;
    font-size:20px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#0284C7;
}

/* Input Labels */
label{
    color:white !important;
    font-weight:bold;
}

/* Metric Cards */
[data-testid="metric-container"]{
    background: linear-gradient(135deg,#2563EB,#06B6D4);
    border-radius:15px;
    padding:20px;
    box-shadow:0px 0px 20px rgba(56,189,248,0.6);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🏦 Loan Eligibility Prediction </div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Loan Approval Prediction</div>", unsafe_allow_html=True)

st.write("")
col1, col2, col3 = st.columns(3)

col1.metric("Models", "3")
col2.metric("Accuracy", "97%")
col3.metric("Status", "Active")

st.write("Enter Customer Financial Details : ")

st.divider()

age = st.number_input("Age",18,60,30)

income = st.number_input(
    "Annual Income",
    value=500000
)

loan = st.number_input(
    "Loan Amount",
    value=200000
)

debt = st.number_input(
    "Existing Debts",
    value=50000
)

utilization = st.slider(
    "Credit Utilization (%)",
    0,
    100,
    30
)

payment = st.slider(
    "Payment History (%)",
    0,
    100,
    90
)

history = st.number_input(
    "Credit History Length (Years)",
    value=5
)

saving = st.number_input(
    "Savings Balance",
    value=150000
)

emi = st.number_input(
    "Monthly EMI",
    value=8000
)

ratio = st.slider(
    "Debt To Income Ratio",
    0,
    100,
    20
)

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "Age": age,
        "Annual_Income": income,
        "Loan_Amount": loan,
        "Existing_Debts": debt,
        "Credit_Utilization": utilization,
        "Payment_History": payment,
        "Credit_History_Length": history,
        "Savings_Balance": saving,
        "EMI_Amount": emi,
        "Debt_to_Income_Ratio": ratio
    }])

    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.divider()

    if prediction == 1:
        st.markdown("""
    <div class="result-box">
        <h2>✅ Customer is Creditworthy</h2>
    </div>
    """, unsafe_allow_html=True)
    else:
        st.markdown("""
    <div style="padding:20px;background:#FEE2E2;
    border-radius:15px;border:2px solid red;">
        <h2>❌ Customer is NOT Creditworthy</h2>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Prediction Probability")

    st.progress(float(probability[1]))

    st.write(f"✅ Creditworthy : {probability[1]*100:.2f}%")

    st.write(f"❌ Not Creditworthy : {probability[0]*100:.2f}%")

    st.subheader("Recommendation")

    if prediction == 1:

        st.success("Loan can be approved.")

        st.write("✔ Good payment history")

        st.write("✔ Maintain current financial behaviour")

    else:

        st.warning("Loan approval is risky.")

        st.write("✔ Improve payment history")

        st.write("✔ Reduce debts")

        st.write("✔ Increase savings")

        st.write("✔ Reduce credit utilization")

st.divider()
