import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os

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

class StockOptionDataset(Dataset):
    def __init__(self, data_dir, ticker, seq_len=15, target_cols=["bid", "ask"]):
        """
        Loads data for a given ticker from its specific file.
        """
        file_path = os.path.join(data_dir, f"option_data_scaled_{ticker}.csv")
        print(f"Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No data file found for ticker {ticker}")
            
        df = pd.read_csv(file_path)
        
        # Define the feature columns (excluding target columns)
        feature_cols = [
            "strike", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume",
            "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d",
            "day_of_week", "day_of_month", "day_of_year"
        ]
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.ticker = ticker

        # Ensure that none of the required columns are missing
        df.dropna(subset=feature_cols + target_cols, inplace=True)
        
        # Concatenate features and targets
        data_np = df[feature_cols + target_cols].to_numpy(dtype=np.float32)
        
        self.data = data_np
        self.n_features = len(feature_cols)
        self.n_targets = len(target_cols)
        self.seq_len = seq_len
        self.n_samples = self.data.shape[0]
        self.max_index = self.n_samples - self.seq_len - 1
        
    def __len__(self):
        return max(0, self.max_index)
    
    def __getitem__(self, idx):
        x_seq = self.data[idx : idx + self.seq_len, :self.n_features]
        y_val = self.data[idx + self.seq_len, self.n_features : self.n_features + self.n_targets]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32) 