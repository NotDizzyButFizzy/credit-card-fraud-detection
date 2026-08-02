"""
generate_sample_data.py
Creates a small SYNTHETIC dataset with the same columns as the real Kaggle
credit-card dataset (Time, V1 to V28, Amount, Class).

Why this exists:
The real dataset is 150 MB and lives on Kaggle behind a login, so it cant be
downloaded inside an automated CI pipeline. This script generates a realistic
stand-in so the whole pipeline can be tested automatically. In a production
setup, this step would instead pull the real data from cloud storage (e.g. S3).

Run it with:  python generate_sample_data.py
"""

import numpy as np
import pandas as pd

COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def generate(n_legit=8000, n_fraud=150, seed=42, path="creditcard.csv"):
    rng = np.random.default_rng(seed)

    # Legitimate and fraudulent transactions come from slightly different
    # distributions, so the model has a real (but not trivial) signal to learn.
    legit = rng.normal(0.0, 1.0, (n_legit, 30))
    fraud = rng.normal(1.6, 1.0, (n_fraud, 30))

    legit_df = pd.DataFrame(legit, columns=COLUMNS)
    legit_df["Time"] = rng.uniform(0, 1.7e5, n_legit)
    legit_df["Amount"] = rng.exponential(70, n_legit)
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud, columns=COLUMNS)
    fraud_df["Time"] = rng.uniform(0, 1.7e5, n_fraud)
    fraud_df["Amount"] = rng.exponential(180, n_fraud)
    fraud_df["Class"] = 1

    df = pd.concat([legit_df, fraud_df]).sample(frac=1, random_state=seed)
    df = df.reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path} (fraud rate {df.Class.mean():.2%})")
    return df


if __name__ == "__main__":
    generate()
