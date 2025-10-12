"""
train_models.py
----------------
This script trains multiple machine learning models (Random Forest, XGBoost,
Decision Tree, and Logistic Regression) on the preprocessed dataset.

It handles:
- Loading preprocessed data
- Splitting data into train/test sets
- Training models
- Evaluating basic performance (accuracy, ROC-AUC)
- Saving trained models for later evaluation

Author: Caxton Henry Matete
"""

# ========== IMPORTS ==========
import os
import pandas as pd
import numpy as np
import joblib  # For saving trained models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# Path to preprocessed file 
DATA_PATH = "data/processed/aml_preprocessed.parquet"

# Directory to save trained models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path):
    """
    Loads the preprocessed dataset.
    Supports both .csv and .parquet formats.
    """
    print("[INFO] Loading preprocessed data...")
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    print(f"[INFO] Dataset loaded successfully with shape: {df.shape}")
    return df


def prepare_features(df):
    """
    Splits dataframe into features (X) and target (y).
    Assumes the target column is named 'label' or 'target'.
    """
    possible_targets = ['label', 'target', 'fraud_label', 'is_fraud']
    target_col = next((col for col in possible_targets if col in df.columns), None)
    
    if target_col is None:
        raise ValueError("Target column not found. Ensure dataset has one of: 'label', 'target', 'fraud_label', or 'is_fraud'.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"[INFO] Features and target separated. Feature shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def evaluate_model(model, X_test, y_test):
    """
    Evaluates model using accuracy and ROC-AUC metrics.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Handle cases where model doesn't support predict_proba (e.g. some linear models)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan
    
    return acc, roc_auc


def train_and_save_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Trains, evaluates, and saves a single model.
    """
    print(f"\n[INFO] Training {model_name}...")
    model.fit(X_train, y_train)
    
    acc, roc_auc = evaluate_model(model, X_test, y_test)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] {model_name} saved to: {model_path}")
    
    # Return performance metrics
    return {
        "model": model_name,
        "accuracy": acc,
        "roc_auc": roc_auc
    }


#MAIN PIPELINE
def main():
    # 1. Load and prepare data
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("[INFO] Data split into train/test sets.")

    # 3. Define models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    #Train and evaluate all models
    results = []
    for name, model in tqdm(models.items(), desc="Training models"):
        res = train_and_save_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)
    
    # 5. Store performance summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_DIR, "model_performance_summary.csv"), index=False)
    print("\n[INFO] Training completed. Performance summary saved.")
    print(results_df)


if __name__ == "__main__":
    main()
