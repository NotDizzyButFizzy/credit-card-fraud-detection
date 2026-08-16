import os

import pandas as pd
import joblib
import streamlit as st

COLS_TO_SCALE = ["Time", "Amount"]


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


@st.cache_data
def load_data():
    for path in ("creditcard.csv", "sample_transactions.csv"):
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("No creditcard.csv or sample_transactions.csv found.")


def score(transaction, model, scaler):
    """Scale one transaction and return (prediction 0/1, fraud probability)."""
    tx = transaction.copy()
    tx[COLS_TO_SCALE] = scaler.transform(tx[COLS_TO_SCALE])
    prediction = int(model.predict(tx)[0])
    fraud_probability = float(model.predict_proba(tx)[0][1])
    return prediction, fraud_probability


st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection")
st.write(
    "An interactive demo of a machine learning model that flags fraudulent "
    "card transactions. Load an example below, adjust it, and watch the "
    "model's verdict change."
)
st.caption(
    "Note: the V1-V28 features are anonymised by the bank, so this runs on "
    "dataset examples, not real card transactions. It demonstrates the model."
)

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Trained model not found. Run `python train.py` first to create it.")
    st.stop()

try:
    df = load_data()
except FileNotFoundError:
    st.error("No data found. Add creditcard.csv (local) or sample_transactions.csv.")
    st.stop()

feature_cols = [c for c in df.columns if c != "Class"]

if "tx" not in st.session_state:
    st.session_state.tx = None

st.subheader("1. Load an example transaction")
col1, col2, col3 = st.columns(3)
if col1.button("Random legitimate"):
    sample = df[df["Class"] == 0].sample(1)
    st.session_state.tx = sample[feature_cols].reset_index(drop=True)
if col2.button("Random fraud"):
    sample = df[df["Class"] == 1].sample(1)
    st.session_state.tx = sample[feature_cols].reset_index(drop=True)
if col3.button("Random (any)"):
    sample = df.sample(1)
    st.session_state.tx = sample[feature_cols].reset_index(drop=True)

if st.session_state.tx is None:
    st.info("Click one of the buttons above to load a transaction.")
    st.stop()

tx = st.session_state.tx.copy()

st.subheader("2. Adjust the transaction")
current_amount = float(tx.loc[0, "Amount"])

max_amount = max(3000.0, round(current_amount + 500, 2))
new_amount = st.slider("Amount (£)", 0.0, max_amount, current_amount)
tx.loc[0, "Amount"] = new_amount


prediction, probability = score(tx, model, scaler)

st.subheader("3. Model verdict")
if prediction == 1:
    st.error(f"FRAUD FLAGGED  —  {probability:.1%} fraud probability")
else:
    st.success(f"Approved  —  {probability:.1%} fraud probability")
st.progress(probability)
st.caption("The bar shows how confident the model is that this is fraud.")

with st.expander("See this transaction's raw feature values"):
    st.dataframe(tx.T, use_container_width=True)
