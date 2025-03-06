import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path

def get_available_tickers(data_dir):
    """Load and return sorted list of available tickers from metadata file."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"\nError: Data directory '{data_dir}' not found!\n"
            f"Please run the split_data.py script first to prepare the data:\n"
            f"  python split_data.py"
        )
        
    metadata_file = os.path.join(data_dir, 'ticker_metadata.csv')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"\nError: Metadata file not found at '{metadata_file}'!\n"
            f"Please run the split_data.py script first to prepare the data:\n"
            f"  python split_data.py"
        )
        
    metadata = pd.read_csv(metadata_file)
    return list(metadata['ticker']), list(metadata['count'])

def select_ticker(tickers, counts):
    """Display tickers with their data counts and get user selection."""
    print("\nAvailable tickers and their data points:")
    print("-" * 50)
    print(f"{'#':<4} {'Ticker':<10} {'Data Points':>12}")
    print("-" * 50)
    
    for i, (ticker, count) in enumerate(zip(tickers, counts), 1):
        print(f"{i:<4} {ticker:<10} {count:>12,}")
    
    while True:
        try:
            choice = input("\nEnter the number of the ticker you want to analyze (or 'q' to quit): ")
            if choice.lower() == 'q':
                exit()
            choice = int(choice)
            if 1 <= choice <= len(tickers):
                selected_ticker = tickers[choice-1]
                selected_count = counts[choice-1]
                print(f"\nSelected ticker: {selected_ticker} ({selected_count:,} data points)")
                return selected_ticker
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

def validate_paths(config):
    """Validate and create necessary directories."""
    data_path = Path(config['data_dir'])
    models_dir = Path(config['models_dir'])
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    models_dir.mkdir(exist_ok=True)
    return data_path, models_dir

class StockOptionDataset(Dataset):
    def __init__(self, data_dir, ticker, seq_len=15, target_cols=["bid", "ask"], 
                 window_sizes=[3, 5, 10], verbose=False):
        """
        Loads data for a given ticker from its specific file and adds rolling window features.
        Rolling operations are skipped for features that are categorical/deterministic such as
        day_of_week, day_of_month, day_of_year, and daysToExpiry.
        """
        file_path = os.path.join(data_dir, f"option_data_scaled_{ticker}.csv")
        if verbose:
            print(f"Loading data from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No data file found for ticker {ticker}")
            
        df = pd.read_csv(file_path)
        
        # Identify numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        if verbose:
            print(f"Excluding non-numeric columns: {non_numeric_cols}")
        
        # Base feature columns (only include existing numeric features)
        base_feature_cols = [
            "strike", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume",
            "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d",
            "day_of_week", "day_of_month", "day_of_year"
        ]
        base_feature_cols = [col for col in base_feature_cols if col in numeric_cols]
        
        # Define features that should NOT be subject to rolling window operations.
        no_rolling_features = {"day_of_week", "day_of_month", "day_of_year", "daysToExpiry"}
        # Create a list of features eligible for rolling computations.
        rolling_base_feature_cols = [col for col in base_feature_cols if col not in no_rolling_features]
        
        # Ensure target columns are numeric
        target_cols = [col for col in target_cols if col in numeric_cols]
        
        # Sort by date if a date column exists
        date_col = None
        for col in ['date', 'Date', 'timestamp', 'Timestamp']:
            if col in df.columns:
                date_col = col
                break
        if date_col:
            df = df.sort_values(date_col).reset_index(drop=True)
        
        # Compute rolling window features for eligible columns only
        rolling_features = {}
        if verbose:
            print("Starting rolling window feature computation...")
        for window in window_sizes:
            if verbose:
                print(f"Processing window size {window}...")
            for idx, col in enumerate(rolling_base_feature_cols):
                if verbose:
                    print(f"  Processing column '{col}' ({idx+1}/{len(rolling_base_feature_cols)}) for window {window}")
                # Ensure enough data points are available
                if df[col].count() > window + 5:
                    rolling_series = df[col].rolling(window)
                    rolling_features[f'{col}_mean_{window}d'] = rolling_series.mean()
                    rolling_features[f'{col}_std_{window}d'] = rolling_series.std()
                    if not (df[col] == 0).all():
                        rolling_features[f'{col}_change_{window}d'] = df[col].pct_change(window)
        
        if rolling_features:
            if verbose:
                print("Concatenating new features to DataFrame...")
            df = pd.concat([df, pd.DataFrame(rolling_features)], axis=1)
            if verbose:
                print("Rolling window feature computation completed.")
        
        # Gather all numeric columns after feature creation, excluding target columns
        feature_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in target_cols]
        
        # Drop rows with NaN values introduced by rolling operations
        df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
        
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.ticker = ticker
        
        # Concatenate features and targets into a numpy array
        data_np = df[self.feature_cols + target_cols].to_numpy(dtype=np.float32)
        
        self.data = data_np
        self.n_features = len(self.feature_cols)
        self.n_targets = len(target_cols)
        self.seq_len = seq_len
        self.n_samples = self.data.shape[0]
        self.max_index = self.n_samples - self.seq_len - 1
        
        if verbose:
            print(f"Created dataset with {self.n_features} features and {self.n_samples} samples")
            print(f"Target columns: {self.target_cols}")
    
    def __len__(self):
        return max(0, self.max_index)
    
    def __getitem__(self, idx):
        x_seq = self.data[idx : idx + self.seq_len, :self.n_features]
        y_val = self.data[idx + self.seq_len, self.n_features : self.n_features + self.n_targets]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
