"""
train.py
--------
Trains the fraud-detection model, prints how well it did, and saves it to disk
so the simulation can reuse it without retraining every time.

Run it with:  python train.py
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from data import load_data, preprocess


def build_model():
    """Create the (untrained) model. One definition, reused everywhere:
    training, validation, and tests all build the model the same way."""
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def main():
    # Load and prepare the data using the functions from data.py
    X, y = load_data()
    X_train, y_train, X_test, y_test, scaler = preprocess(X, y)

    # Train the model. Random Forest is a solid, well-understood choice.
    print("Training model...")
    model = build_model()
    model.fit(X_train, y_train)

    # Evaluate on the untouched test set.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Results on the test set ---")
    # digits=4 gives us precise precision/recall to copy into the README.
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Save the model AND the scaler. We need both to score live transactions,
    # because a new transaction must be scaled the same way as the training data.
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("\nSaved model and scaler to the models/ folder.")


if __name__ == "__main__":
    main()
