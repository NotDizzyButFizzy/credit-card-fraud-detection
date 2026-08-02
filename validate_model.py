"""
validate_model.py
-----------------
A model validation "gate" for the retraining pipeline.

It retrains the model and checks it clears a minimum performance bar before the
model is allowed through. If the retrained model is worse than the bar, this
script exits with an error, which makes the CI/CD pipeline fail and stops a bad
model from being accepted.

This is the honest core of "continuous deployment for model retraining":
retrain -> validate against a quality bar -> only accept if it passes.

Run it with:  python validate_model.py
"""

import os
import sys

import joblib
from sklearn.metrics import recall_score, precision_score

from data import load_data, preprocess
from train import build_model

# The minimum performance we will accept. Recall matters most for fraud:
# a missed fraud costs the bank money, so we refuse to accept a model that
# catches too little fraud, even if it looks accurate overall.
MIN_RECALL = 0.70
MIN_PRECISION = 0.50


def validate(path="creditcard.csv", save=False):
    X, y = load_data(path)
    X_train, y_train, X_test, y_test, scaler = preprocess(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    print(f"Recall on fraud:    {recall:.4f}  (minimum {MIN_RECALL})")
    print(f"Precision on fraud: {precision:.4f}  (minimum {MIN_PRECISION})")

    if recall < MIN_RECALL or precision < MIN_PRECISION:
        print("\nVALIDATION FAILED: model does not meet the quality bar.")
        return False

    print("\nVALIDATION PASSED: model meets the quality bar.")

    # Only a model that passed the gate is allowed to be saved/accepted.
    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fraud_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        print("Accepted model saved to models/.")

    return True


if __name__ == "__main__":
    # Exit code 1 tells the CI/CD pipeline the step failed.
    sys.exit(0 if validate(save=True) else 1)
