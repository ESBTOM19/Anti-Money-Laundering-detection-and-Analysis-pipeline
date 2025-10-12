"""
visualize_results.py
---------------------
This script visualizes model evaluation results and feature importances.

It performs:
- Visualization of key performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Comparison plots across models
- Feature importance visualization for tree-based models (if available)
- Correlation heatmaps and summary insights

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8')

MODEL_DIR = "models"
DATA_PATH = "data/processed/aml_preprocessed.parquet"
RESULTS_PATH = os.path.join(MODEL_DIR, "evaluation_results.csv")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results():
    """Loads evaluation metrics saved by evaluate_models.py."""
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Results file not found at {RESULTS_PATH}. Run evaluate_models.py first.")
    
    results = pd.read_csv(RESULTS_PATH)
    print(f"[INFO] Loaded evaluation results with shape: {results.shape}")
    print(results)
    return results


def plot_metric_comparison(results, metric):
    """Plots comparison of a given metric across all models."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x="model", y=metric, data=results, palette="viridis")
    plt.title(f"Model Comparison: {metric.capitalize()}")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{metric}_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✔ Saved {metric} comparison chart to {path}")


def plot_all_metrics(results):
    """Generates comparison plots for all major metrics."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    for metric in metrics:
        if metric in results.columns:
            plot_metric_comparison(results, metric)


def plot_feature_importance():
    """Plots feature importances for RandomForest, XGBoost, and DecisionTree if available."""
    tree_models = ["RandomForest", "XGBoost", "DecisionTree"]
    for file in os.listdir(MODEL_DIR):
        for name in tree_models:
            if name.lower() in file.lower() and file.endswith(".pkl"):
                model_path = os.path.join(MODEL_DIR, file)
                model = joblib.load(model_path)
                if hasattr(model, "feature_importances_"):
                    print(f"[INFO] Plotting feature importance for {name}...")
                    # Try to load the data to get feature names
                    df = pd.read_parquet(DATA_PATH)
                    X = df.drop(columns=[col for col in ['label', 'target', 'fraud_label', 'is_fraud'] if col in df.columns])
                    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    
                    plt.figure(figsize=(8, 6))
                    importances.head(15).plot(kind='barh', color='teal')
                    plt.title(f"Top 15 Feature Importances - {name}")
                    plt.xlabel("Importance Score")
                    plt.tight_layout()
                    path = os.path.join(PLOTS_DIR, f"{name}_feature_importance.png")
                    plt.savefig(path)
                    plt.close()
                    print(f"Saved {name} feature importance plot to {path}")


def plot_correlation_heatmap():
    """Plots correlation matrix of numerical features in the dataset."""
    print("[INFO] Generating correlation heatmap...")
    df = pd.read_parquet(DATA_PATH)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved correlation heatmap to {path}")


def summarize_results(results):
    """Prints key takeaways based on the evaluation metrics."""
    best_model = results.sort_values(by="roc_auc", ascending=False).iloc[0]
    print("\nSUMMARY INSIGHTS")
    print(f"Best Overall Model: {best_model['model']}")
    print(f"Accuracy: {best_model['accuracy']:.3f} | Precision: {best_model['precision']:.3f} | Recall: {best_model['recall']:.3f}")
    print(f"F1 Score: {best_model['f1_score']:.3f} | ROC AUC: {best_model['roc_auc']:.3f}")
    print("\nInsights:")
    print("- High Recall → fewer missed suspicious transactions.")
    print("- High Precision → fewer false alarms for investigators.")
    print("- Balance Recall & Precision depending on operational priority.")
    print("======================================")


def main():
    print("========== MODEL VISUALIZATION STARTED ==========")
    results = load_results()
    
    # Generate metric comparison visuals
    print("[INFO] Generating metric comparison plots...")
    plot_all_metrics(results)
    
    # Feature importance for interpretable models
    print("[INFO] Checking and plotting feature importances...")
    plot_feature_importance()
    
    # Correlation heatmap for data insight
    print("[INFO] Plotting feature correlation heatmap...")
    plot_correlation_heatmap()

    # Summary insights
    summarize_results(results)
    print("\n========== VISUALIZATION COMPLETE ==========")


if __name__ == "__main__":
    main()
