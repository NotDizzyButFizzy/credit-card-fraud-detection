Credit Card Fraud Detection
A machine learning project that detects fraudulent credit card transactions using a Random Forest classifier, trained on a highly imbalanced real-world dataset.

Overview
Credit card fraud is a significant financial problem, with fraudulent transactions making up only a tiny fraction of all activity. This project explores that imbalance, analyses transaction patterns, and builds a classification model capable of flagging fraud with high accuracy.
The notebook walks through the full data science pipeline: loading and exploring the data, visualising class distribution and feature correlations, training a Random Forest model, and evaluating it with a comprehensive set of metrics.

Dataset
This project uses the Credit Card Fraud Detection dataset from Kaggle.

284,807 transactions made by European cardholders
492 fraudulent transactions (~0.17% of all records)
Features V1–V28 are PCA-transformed for confidentiality
Additional features: Time, Amount, and Class (0 = legitimate, 1 = fraud)


Download creditcard.csv from Kaggle and update the file path in the notebook accordingly.


Project Structure
├── Credit_card_fraud_detection.ipynb   # Main Jupyter notebook
├── Credit_card_fraud_detection.html    # Rendered HTML export of the notebook
└── README.md

Workflow

Data Loading & Exploration — Load the CSV, inspect the first rows, and summarise statistics
Class Imbalance Analysis — Quantify the fraud-to-legitimate transaction ratio
Amount Analysis — Compare transaction amounts between fraudulent and valid cases
Correlation Heatmap — Visualise feature relationships using Seaborn
Train/Test Split — 80/20 split with a fixed random seed for reproducibility
Model Training — Random Forest Classifier fitted on the training set
Evaluation — Accuracy, Precision, Recall, F1-Score, Matthews Correlation Coefficient, and a Confusion Matrix heatmap


Requirements
Install dependencies with:
bashpip install pandas scikit-learn matplotlib seaborn numpy
LibraryPurposepandasData loading and manipulationnumpyNumerical operationsscikit-learnModel training and evaluationmatplotlib / seabornData visualisation

Usage

Clone this repository
Download creditcard.csv from Kaggle
Update the file path in the notebook:

python   data = pd.read_csv("path/to/creditcard.csv")

Launch Jupyter and run all cells:

bash   jupyter notebook Credit_card_fraud_detection.ipynb

Results
The model is evaluated using metrics suited to imbalanced classification:
MetricDescriptionAccuracyOverall correct predictionsPrecisionOf flagged frauds, how many were realRecallOf real frauds, how many were caughtF1-ScoreHarmonic mean of precision and recallMCCBalanced measure accounting for all four confusion matrix values
A confusion matrix is plotted at the end of the notebook for a visual breakdown of predictions.

Notes

The dataset is heavily imbalanced. MCC and F1-Score are more meaningful indicators of model performance than raw accuracy.
No sampling techniques (e.g. SMOTE) are applied in this version — a good avenue for further experimentation.
The Random Forest is used with default hyperparameters; tuning via GridSearchCV could improve recall on the minority class.
