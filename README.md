# Credit Card Fraud Detection

[![CI](https://github.com/NotDizzyButFizzy/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/NotDizzyButFizzy/credit-card-fraud-detection/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](REPLACE_WITH_YOUR_STREAMLIT_LINK)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

A machine learning system that identifies fraudulent credit card transactions in a highly imbalanced dataset — wrapped in production-style engineering: modular code, a real-time scoring simulation, automated tests, a CI/CD pipeline, and an interactive web app.

**▶ Try the live demo:** REPLACE_WITH_YOUR_STREAMLIT_LINK — load a transaction and watch the model flag fraud in real time.

---

## Overview

Fraud detection is a needle-in-a-haystack problem. In this dataset, fewer than **0.2%** of transactions are fraudulent, which means a model that blindly labels everything "legitimate" would still be 99.8% accurate — while catching zero fraud. Accuracy is a trap here.

This project tackles that head-on. It handles the class imbalance with SMOTE, trains a Random Forest classifier, and is judged on the metrics that actually matter for a bank: **recall** (how much fraud is caught) and **precision** (how few false alarms are raised). Around the model sits the kind of software engineering used in real production systems — clean, modular code, a live simulation, an automated test suite, a CI/CD workflow, and an interactive web app deployed for anyone to try.

---

## Technologies Used

- **Python 3.11** — core language
- **pandas / NumPy** — data manipulation and numerical work
- **scikit-learn** — model training and evaluation
- **imbalanced-learn (SMOTE)** — handling class imbalance
- **Matplotlib / Seaborn** — data visualisation
- **joblib** — saving and loading the trained model
- **Streamlit** — the interactive web app
- **pytest / flake8** — automated testing and code linting
- **GitHub Actions** — continuous integration and deployment
- **Jupyter Notebook** — exploratory data analysis

---

## Features

- **Handles severe class imbalance** with SMOTE, applied to the training data only to avoid data leakage.
- **Random Forest classifier** trained and evaluated on unseen data.
- **Evaluation focused on recall, precision and F1** rather than misleading accuracy.
- **Interactive web app** — deployed live, so anyone can load a transaction and see the model's verdict in their browser.
- **Real-time transaction simulation** that streams transactions through the model one at a time and flags fraud live.
- **Modular codebase** — data preparation, training, and simulation are cleanly separated.
- **Automated test suite** run on every push through continuous integration.
- **Model retraining pipeline** with a validation gate that rejects any model failing to clear a minimum recall and precision bar.
- **Fully reproducible** — one command installs everything from `requirements.txt`.

---

## The Process

1. **Data exploration** — examined the dataset, confirmed the extreme class imbalance, and looked at how fraudulent and legitimate transactions differ.
2. **Preprocessing** — split the data into training and test sets first, scaled `Time` and `Amount`, and balanced the training set with SMOTE (leaving the test set untouched so evaluation stays realistic).
3. **Model training** — trained a Random Forest classifier on the balanced data.
4. **Evaluation** — measured recall, precision, F1 and ROC-AUC on the untouched test set, and inspected the confusion matrix.
5. **Simulation** — built a real-time simulation that loads the trained model and scores transactions one by one, as a payment system would.
6. **Engineering** — refactored the workflow into modules, added an automated test suite, and set up a CI/CD pipeline with GitHub Actions.
7. **Deployment** — built an interactive web app with Streamlit and deployed it publicly, so the model can be tried live in the browser.

---

## Results

Measured on the held-out test set, using the full pipeline (with SMOTE). Because the data is imbalanced, the focus is on the **fraud class**, not overall accuracy.

| Metric | Score |
| --- | --- |
| Precision (fraud) | 0.8454 |
| Recall (fraud) | 0.8367 |
| F1 (fraud) | 0.8410 |
| ROC-AUC | 0.9731 |

**In plain terms:** the model catches roughly **84% of all fraud** while keeping false alarms low. Balancing the training data with SMOTE deliberately trades a little precision for higher recall — catching more fraud — which is usually the right call for a bank, where a missed fraud tends to cost more than a declined-then-verified card.

---

## Project Structure

```
credit-card-fraud-detection/
├── .github/workflows/
│   ├── ci.yml                  # CI: lint + test on every push
│   └── retrain.yml             # Retrain + validate the model on demand/schedule
├── tests/
│   └── test_pipeline.py        # Automated test suite
├── app.py                      # Interactive Streamlit web app
├── data.py                     # Load and preprocess (split, scale, SMOTE)
├── train.py                    # Train the model, evaluate, save it
├── simulate.py                 # Real-time transaction simulation
├── validate_model.py           # Quality gate: accept a model only if it passes
├── generate_sample_data.py     # Synthetic stand-in data for CI
├── prepare_deploy.py           # Builds the small model + sample for deployment
├── sample_transactions.csv     # Small sample the web app uses for examples
├── models/                     # Trained model and scaler
├── conftest.py                 # Test configuration helper
├── requirements.txt            # Dependencies
├── .flake8                     # Linter configuration
└── credit_card_fraud_detection.ipynb   # Exploratory analysis
```

---

## How to Run

### Run the web app

```bash
pip install -r requirements.txt
python train.py                 # creates the model (needs creditcard.csv)
streamlit run app.py            # launches the app in your browser
```

### Run the pipeline directly

```bash
python train.py                 # trains the model and prints the results
python simulate.py              # streams transactions through the model live
```

The full dataset is not included (it is ~150MB). Download it from Kaggle and place
`creditcard.csv` in the project folder:
[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

To explore the analysis and charts, open `credit_card_fraud_detection.ipynb` in Jupyter.

---

## Live Demo

The web app lets anyone try the model without any code. Load an example transaction
(legitimate, fraud, or random), and the model gives a live verdict — **fraud** or
**approved** — along with how confident it is. A slider lets you adjust the amount
and watch the prediction respond.

**Note:** the `V1`–`V28` features are anonymised by the bank, so the app runs on
dataset examples, not real card transactions. It demonstrates how the model works.

---

## Continuous Integration / Continuous Deployment

This project uses **GitHub Actions** for CI/CD:

- **`ci.yml`** runs automatically on every push and pull request. It spins up a clean machine, installs the project, lints the code with flake8, and runs the full test suite with pytest — so any change that breaks the code or fails a test is caught immediately.
- **`retrain.yml`** is the model-retraining pipeline. It retrains the model and runs it through a validation gate (`validate_model.py`), which only accepts the model if it clears a minimum recall and precision bar. A worse model causes the pipeline to fail, so it can never be silently accepted.

---

## What I Learnt

- **Why accuracy is misleading on imbalanced data**, and how to read recall, precision, F1 and a confusion matrix instead.
- **How to handle class imbalance with SMOTE** — and why it must only be applied to the training data to avoid leaking information into evaluation.
- **The precision/recall trade-off** in fraud detection, and how it's ultimately a business decision about the relative cost of missed fraud versus false alarms.
- **How to structure a project into modules** so the code is reusable, testable, and easy to maintain.
- **How to write automated tests** and wire them into a **CI/CD pipeline** with GitHub Actions.
- **How to deploy a model as a live web app** that anyone can use.

---

## How It Could Be Improved

- Compare additional models (e.g. gradient boosting / XGBoost, logistic regression) against the Random Forest baseline.
- Deliberately tune the decision threshold to push recall higher, and report the precision cost.
- Add a with-versus-without-SMOTE comparison to demonstrate its effect on recall.
- Deploy to a scalable cloud setup (e.g. AWS, behind an API) to serve predictions at higher volume.
- Add monitoring for data drift, so the model is retrained when incoming transactions start to differ from the training data.
