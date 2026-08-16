"""
Automated tests for the fraud-detection pipeline.

These run on small synthetic data (no Kaggle download needed), so they are fast
and can run automatically in CI on every push. They check the important
behaviours: preprocessing balances the data without touching the test set, the
model produces valid predictions, the live scorer returns sensible output, and
the validation gate accepts a good model.
"""

import numpy as np
import pandas as pd

from data import preprocess
from train import build_model
from simulate import score_transaction

COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def make_data(n_legit=400, n_fraud=40, seed=0):
    """Build a small in-memory dataset shaped like the real one."""
    rng = np.random.default_rng(seed)
    legit = rng.normal(0.0, 1.0, (n_legit, 30))
    fraud = rng.normal(1.6, 1.0, (n_fraud, 30))

    legit_df = pd.DataFrame(legit, columns=COLUMNS)
    legit_df["Time"] = rng.uniform(0, 1e5, n_legit)
    legit_df["Amount"] = rng.exponential(70, n_legit)
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud, columns=COLUMNS)
    fraud_df["Time"] = rng.uniform(0, 1e5, n_fraud)
    fraud_df["Amount"] = rng.exponential(180, n_fraud)
    fraud_df["Class"] = 1

    df = pd.concat([legit_df, fraud_df]).sample(frac=1, random_state=seed)
    df = df.reset_index(drop=True)
    return df.drop("Class", axis=1), df["Class"]


def test_preprocess_balances_training_data():
    """SMOTE should make the two classes equal in the TRAINING set."""
    X, y = make_data()
    _, y_train, _, _, _ = preprocess(X, y, test_size=0.2, random_state=0)
    counts = np.bincount(np.asarray(y_train))
    assert counts[0] == counts[1]


def test_preprocess_leaves_test_set_size_correct():
    """The test set should be the expected fraction and stay untouched."""
    X, y = make_data()  # 440 rows -> 20% = 88 rows
    _, _, X_test, _, _ = preprocess(X, y, test_size=0.2, random_state=0)
    assert len(X_test) == 88
    assert list(X_test.columns) == list(X.columns)


def test_scaler_is_fitted():
    """The returned scaler must have been fitted (so we can reuse it live)."""
    X, y = make_data()
    _, _, _, _, scaler = preprocess(X, y, random_state=0)
    assert hasattr(scaler, "mean_")


def test_model_predicts_binary_labels():
    """The trained model should output only 0/1 labels, one per row."""
    X, y = make_data()
    X_train, y_train, X_test, _, _ = preprocess(X, y, random_state=0)
    model = build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_score_transaction_returns_valid_output():
    """The live scorer should return a 0/1 label and a probability in [0, 1]."""
    X, y = make_data()
    X_train, y_train, _, _, scaler = preprocess(X, y, random_state=0)
    model = build_model()
    model.fit(X_train, y_train)

    one_transaction = X.iloc[[0]]  # a single raw (unscaled) transaction
    prediction, probability = score_transaction(one_transaction, model, scaler)
    assert prediction in (0, 1)
    assert 0.0 <= probability <= 1.0


def test_validation_gate_accepts_good_model(tmp_path):
    """End-to-end: generated data should train a model that passes the gate."""
    import generate_sample_data
    import validate_model

    data_path = tmp_path / "creditcard.csv"
    generate_sample_data.generate(n_legit=1500, n_fraud=60, path=str(data_path))
    assert validate_model.validate(path=str(data_path)) is True
