Credit Card Fraud Detection
My first ML project - from Jupyter struggles to 99.93% accuracy in 24 hours

The Challenge

284,807 credit card transactions with extreme imbalance:
    492 fraud (0.17%) vs 284,315 normal
    Goal: Catch fraud without too many false alarms
    
My Journey
Day 1 struggles → Production ML:
❌ "externally-managed-environment" → venv mastery
❌ NameError: 'X_train not defined' → variable debugging
❌ Kernel crashes after pip install → restart pro
✅ Scaling + RandomForest = 99.9% accuracy

Took 24 hours of debugging = real engineering skills
Results That Matter
| Metric    | Score  | Why it matters         |
| --------- | ------ | ---------------------- |
| Accuracy  | 99.93% | Overall performance    |
| Precision | 92.86% | Few false fraud alarms |
| Recall    | 82.35% | Catches 82% real fraud |
| F1-Score  | 87.27% | Balance of both        |
| MCC       | 0.8704 | Best imbalanced metric |

Key Insights
  V17, V14, V12 = top fraud indicators (correlation heatmap)
  Fraud transactions average $122 vs normal $88
  High precision = bank-friendly (few customer complaints)
  82% recall = catches most fraudsters

Dataset
  Kaggle Credit Card Fraud Detection https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
  284k transactions - 31 features - PCA anonymized - 150MB CSV

Tech stack
Jupyter Lab + Python venv
pandas • scikit-learn • RandomForestClassifier(100)
matplotlib • seaborn • Confusion Matrix • MCC

requirements.txt
pandas
scikit-learn
matplotlib
seaborn
numpy
imbalanced-learn
