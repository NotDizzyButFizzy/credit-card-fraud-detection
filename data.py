@@ -1,56 +1,55 @@
"""
data.py
-------
Everything to do with loading and preparing the data lives here.

Keeping this separate means train.py and simulate.py can both reuse the exact
same preprocessing without copying and pasting it.
same preprocesing without copying and pasting it.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# The two columns that are on a "raw" scale and need standardising.
# V1..V28 are already scaled (they come from PCA), so we leave them alone.
COLS_TO_SCALE = ["Time", "Amount"]


def load_data(path="creditcard.csv"):
    """Load the CSV and split it into features (X) and the target label (y)."""
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)   # everything except the label
    y = df["Class"]                # 0 = legitimate, 1 = fraud
    return X, y


def preprocess(X, y, test_size=0.2, random_state=42):
    """
    Prepare the data for training, in the correct order to avoid data leakage:

      1. Split into train/test FIRST (so the test set stays untouched).
      2. Fit the scaler on the training data only, then apply it to both.
      3. Balance the TRAINING data only, using SMOTE.

    Returns the balanced training set, the untouched test set, and the fitted
    scaler (we need the scaler later to preprocess live transactions).
    """
    # 1. Split first. stratify=y keeps the same fraud ratio in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Scale. Fit on train, then transform both. Never fit on the test set.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])

    # 3. Balance the training data. Fraud is ~0.2% of rows, so we create
    #    synthetic fraud examples with SMOTE so the model has enough to learn from.
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced, X_test, y_test, scaler
