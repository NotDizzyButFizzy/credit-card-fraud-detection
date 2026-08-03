"""
prepare_deploy.py
-----------------
Creates the small files needed to deploy the app to a public link, since the
full 150MB dataset is too big for GitHub.

It produces:
  1. sample_transactions.csv  - a small sample of transactions for the app's
     example buttons to use.
  2. models/fraud_model.pkl + models/scaler.pkl - a compressed trained model.

Run it locally once (you need creditcard.csv present):  python prepare_deploy.py
"""

import os
import joblib
import pandas as pd

from data import load_data, preprocess
from train import build_model


def main():
    X, y = load_data()

    # 1. A small, balanced sample for the app's example buttons.
    df = pd.concat([X, y], axis=1)
    n_fraud = min(150, int((df["Class"] == 1).sum()))
    frauds = df[df["Class"] == 1].sample(n_fraud, random_state=1)
    legits = df[df["Class"] == 0].sample(300, random_state=1)
    sample = pd.concat([frauds, legits]).sample(frac=1, random_state=1)
    sample = sample.reset_index(drop=True)
    sample.to_csv("sample_transactions.csv", index=False)
    print(f"Wrote sample_transactions.csv ({len(sample)} rows)")

    # 2. Train and save a COMPRESSED model, so the file is small enough for GitHub.
    X_train, y_train, X_test, y_test, scaler = preprocess(X, y)
    model = build_model()
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.pkl", compress=3)
    joblib.dump(scaler, "models/scaler.pkl", compress=3)

    size_mb = os.path.getsize("models/fraud_model.pkl") / 1e6
    print(f"Saved compressed model ({size_mb:.1f} MB)")
    if size_mb > 90:
        print("WARNING: the model file is close to GitHub's 100MB limit.")
        print("If it won't upload, lower n_estimators in train.py's build_model().")


if __name__ == "__main__":
    main()
