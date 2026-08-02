"""
simulate.py
-----------
A real-time transaction simulation.

It loads the trained model, then streams transactions through it one at a time
with a short delay, printing a live decision for each one -- like a simplified
version of what a payment system does when a card is used.

It is a SIMULATION: the transactions come from the dataset, not a live payment
feed, but each one is scored in real time by the model exactly as it would be
in production.

Run it with:  python simulate.py   (make sure you have run train.py first)
"""

import time
import joblib
import pandas as pd

COLS_TO_SCALE = ["Time", "Amount"]


def load_artifacts():
    """Load the trained model and the scaler saved by train.py."""
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


def score_transaction(transaction, model, scaler):
    """
    Score a single transaction (a one-row DataFrame of raw features).
    Returns the prediction (0/1) and the model's fraud probability.
    """
    tx = transaction.copy()
    # Scale the same columns, the same way, as during training.
    tx[COLS_TO_SCALE] = scaler.transform(tx[COLS_TO_SCALE])
    prediction = model.predict(tx)[0]
    fraud_probability = model.predict_proba(tx)[0][1]
    return prediction, fraud_probability


def run_simulation(path="creditcard.csv", n_fraud=5, n_legit=15, delay=1.0):
    """Stream a shuffled mix of transactions through the model, one at a time."""
    model, scaler = load_artifacts()
    df = pd.read_csv(path)

    # Deliberately include some real fraud so the simulation is interesting.
    frauds = df[df["Class"] == 1].sample(n_fraud, random_state=1)
    legits = df[df["Class"] == 0].sample(n_legit, random_state=1)
    stream = pd.concat([frauds, legits]).sample(frac=1, random_state=1).reset_index(drop=True)

    features = stream.drop(columns="Class")
    labels = stream["Class"]

    print("Incoming transaction stream:\n")
    for i in range(len(stream)):
        transaction = features.iloc[[i]]      # keep it as a one-row DataFrame
        actual = int(labels.iloc[i])
        prediction, prob = score_transaction(transaction, model, scaler)

        status = "FRAUD FLAGGED" if prediction == 1 else "approved"
        amount = features.iloc[i]["Amount"]
        note = "   <-- was actually fraud" if actual == 1 else ""
        print(f"[{i + 1:02d}] Amount: {amount:9.2f} | "
              f"fraud probability: {prob:6.1%} | {status}{note}")

        time.sleep(delay)   # pause so it feels like a live feed

    print("\nStream ended.")


if __name__ == "__main__":
    run_simulation()
