# preprocess.py
"""
This script handles all preprocessing tasks for the Anti-Money-Laundering pipeline:
- Loads synthetic transactions and alerts data
- Converts CSV files to Parquet for faster processing
- Performs feature engineering
- Merges datasets and prepares features (X) and labels (y)
- Saves processed data for modeling
"""

import pandas as pd
from pathlib import Path

#Define directories and paths
data_dir = Path("data")                      # Main data directory
processed_dir = data_dir / "processed"       # Directory to save processed data
processed_dir.mkdir(parents=True, exist_ok=True)  # Create if not exist

transactions_csv = data_dir / "synthetic_transactions.csv"
alerts_csv = data_dir / "synthetic_alerts.csv"

#Function to convert CSV to Parquet
def convert_csv_to_parquet(csv_path, name, sample_size=100_000):
    """
    Converts a CSV to Parquet and creates a smaller sample for testing
    Arguments:
        csv_path (Path): Path to CSV file
        name (str): Base name for output files
        sample_size (int): Number of rows for sample Parquet
    """
    print(f"Processing {csv_path.name} ...")

    #Step 1: Create a smaller sample parquet
    df_sample = pd.read_csv(csv_path, nrows=sample_size)
    df_sample.to_parquet(processed_dir / f"{name}_sample.parquet",
                         index=False, compression="snappy")

    #Step 2: Convert full CSV to Parquet in chunks
    print(f"Converting full {csv_path.name} to Parquet (may take time)...")
    chunksize = 200_000
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        chunk.to_parquet(processed_dir / f"{name}_part{i}.parquet",
                         index=False, compression="snappy")

    print(f"Finished processing {csv_path.name}\n")

#Load and preprocess datasets
def load_and_merge_data():
    """
    Loads the sample Parquet files, merges transactions with alerts,
    and prepares features (X) and target (y)
    """
    # Load samples to avoid memory issues
    transactions = pd.read_parquet(processed_dir / "synthetic_transactions_sample.parquet")
    alerts = pd.read_parquet(processed_dir / "synthetic_alerts_sample.parquet")

    print(f"Loaded {transactions.shape[0]} transactions and {alerts.shape[0]} alerts.")

    # Merge transactions and alerts on AlertID
    merged = transactions.merge(alerts, on="AlertID", how="inner")
    print(f"Merged dataset shape: {merged.shape}")

    # Convert Outcome to binary label: 'Report' = 1, 'Dismiss' = 0
    merged['Label'] = merged['Outcome'].map({'Report': 1, 'Dismiss': 0})

    # Select features and target
    X = merged[['Size']]  # Add more engineered features
    y = merged['Label']

    return X, y

# 4. Save processed features and labels
def save_processed_data(X, y):
    """
    Saves processed features (X) and target (y) as Parquet files
    """
    X.to_parquet(processed_dir / "X.parquet", index=False)
    y.to_parquet(processed_dir / "y.parquet", index=False)
    print(f"Processed data saved to '{processed_dir}'")

# 5.Main execution
if __name__ == "__main__":
    # Convert raw CSVs to Parquet (sample + full chunks)
    convert_csv_to_parquet(transactions_csv, "synthetic_transactions")
    convert_csv_to_parquet(alerts_csv, "synthetic_alerts")

    # Load merged data and prepare X, y
    X, y = load_and_merge_data()

    # Save processed features for modeling
    save_processed_data(X, y)

    print("Preprocessing complete. Ready for modeling.")
