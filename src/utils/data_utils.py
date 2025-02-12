import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

def get_available_tickers(csv_file):
    """Load and return sorted list of unique tickers with their data counts."""
    df = pd.read_csv(csv_file)
    ticker_counts = df['ticker'].value_counts().sort_index()
    return list(ticker_counts.index), list(ticker_counts.values)

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
    def __init__(self, csv_file, ticker="CGC", seq_len=15, target_cols=["bid", "ask"]):
        """
        Loads data for a given ticker and sets up the features and targets.
        
        IMPORTANT CHANGES:
          - The feature columns have been re-ordered to *exclude* the target columns.
          - The target columns are now a list (e.g. ["bid", "ask"]).
        """
        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        
        df = df[df['ticker'] == ticker].copy()
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # Define the feature columns (note: 'bid' and 'ask' have been removed)
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
        
        # Concatenate features and targets (features first, then targets)
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
        # Get a sequence of features
        x_seq = self.data[idx : idx + self.seq_len, :self.n_features]
        # Get the target(s) from the next row; now a vector (e.g. [bid, ask])
        y_val = self.data[idx + self.seq_len, self.n_features : self.n_features + self.n_targets]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32) 