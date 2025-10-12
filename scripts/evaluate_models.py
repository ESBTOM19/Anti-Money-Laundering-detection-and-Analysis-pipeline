"""
evaluate_models.py
------------------
This script evaluates all trained models on the test dataset.

It performs:
- Model loading (from the 'models/' directory)
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Confusion matrix visualization
- ROC curve visualization for all models
- Exports performance summary to CSV and saves plots

"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8')

# ========== CONFIGURATION ==========
MODEL_DIR = "models"
DATA_PATH = "data/processed/aml_preprocessed.parquet"
RESULTS_PATH = os.path.join(MODEL_DIR, "evaluation_results.csv")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


#HELPER FUNCTIONS
def load_data(path):
    """Loads test data from preprocessed dataset."""
    print("[INFO] Loading dataset for evaluation...")
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    print(f"[INFO] Dataset loaded successfully with shape: {df.shape}")
    return df


def prepare_features(df):
    """Splits dataframe into X and y."""
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
    Evaluates a trained model on test data and returns key metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    # Compute ROC-AUC if possible
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = np.nan

    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Generates and saves confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()


def plot_roc_curves(models, X_test, y_test):
    """Plots ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves_all_models.png"))
    plt.close()


#MAIN PIPELINE
def main():
    print("========== MODEL EVALUATION STARTED ==========")
    
    # Load dataset and split
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load models
    models = {}
    print("\n[INFO] Loading trained models...")
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            model_path = os.path.join(MODEL_DIR, file)
            models[model_name] = joblib.load(model_path)
            print(f"  ✔ Loaded {model_name}")

    if not models:
        raise FileNotFoundError("No models found in 'models/' directory.")

    print("\n[INFO] Evaluating models...")
    results = []

    for name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        
        # Save confusion matrix
        plot_confusion_matrix(y_test, y_pred, name)
        print(f"  ✅ {name} evaluated.")

    # Save overall ROC curve comparison
    plot_roc_curves(models, X_test, y_test)

    # Save metrics summary
    results_df = pd.DataFrame(results)
    results_df = results_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    results_df.to_csv(RESULTS_PATH, index=False)
    print("\n[INFO] Evaluation completed. Results saved at:", RESULTS_PATH)
    print(results_df)
    print("MODEL EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
