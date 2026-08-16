import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


COLS_TO_SCALE = ["Time", "Amount"]


def load_data(path="creditcard.csv"):
    """Load the CSV and split it into features (X) and the target label (y)."""
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)   
    y = df["Class"]                
    return X, y


def preprocess(X, y, test_size=0.2, random_state=42):
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])



    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced, X_test, y_test, scaler
