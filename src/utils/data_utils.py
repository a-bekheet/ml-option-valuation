import os
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

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
                 window_sizes=[3, 5, 10], verbose=False,
                 include_greeks: bool = True,
                 include_rolling_features: bool = True,
                 # --- NEW FLAG ---
                 include_cyclical_features: bool = True):
        """
        Loads data for a given ticker, optionally adds rolling window features,
        Greeks, and selects between original or cyclical date features.

        Args:
            data_dir (str): Base directory containing the 'by_ticker' subdirectory.
            ticker (str): The stock ticker symbol.
            seq_len (int): Length of the input sequence.
            target_cols (list): List of target column names.
            window_sizes (list): List of window sizes for rolling features.
            verbose (bool): If True, print detailed loading information.
            include_greeks (bool): If True, include calculated Greek columns as features.
            include_rolling_features (bool): If True, calculate and include rolling window features.
        """
        base_path = Path(data_dir)
        file_path = base_path / 'by_ticker' / f"{ticker}_normalized.csv"

        if verbose:
            logging.info(f"Initializing StockOptionDataset for ticker: {ticker}")
            logging.info(f"Attempting to load data from Colab path: {file_path}")
            logging.info(f"Sequence length: {seq_len}, Target columns: {target_cols}")
            logging.info(f"Include Greeks: {include_greeks}")
            logging.info(f"Include Rolling Features: {include_rolling_features}")
            logging.info(f"Include Cyclical Date Features: {include_cyclical_features}") # Log new flag

        if not file_path.exists():
            logging.error(f"No data file found for ticker {ticker} at {file_path}")
            raise FileNotFoundError(f"No data file found for ticker {ticker} at {file_path}")

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                 logging.warning(f"Data file for ticker {ticker} is empty."); self._set_empty_attributes(target_cols, seq_len, ticker); return
        except Exception as e: logging.error(f"Error loading data for {ticker}: {e}"); raise

        # --- Feature Definition ---
        # Base features EXCLUDING raw date parts
        base_numeric_features = [
            "strike", "lastPrice", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume",
            "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d", "risk_free_rate"
            # Removed 'day_of_week', 'day_of_month', 'day_of_year' from base
        ]
        greek_columns = ["delta", "gamma", "vega", "theta", "rho"]
        # Define date feature sets
        cyclical_date_features = [ "day_of_week_sin", "day_of_week_cos", "day_of_month_sin", "day_of_month_cos", "day_of_year_sin", "day_of_year_cos" ]
        original_date_features = [ "day_of_week", "day_of_month", "day_of_year" ]

        # --- Rolling Window Feature Calculation (Conditional) ---
        rolling_features_data = {}
        if include_rolling_features:
            if verbose: logging.info("Starting rolling window feature computation...")
            # Logic remains the same, ensure it doesn't roll date features if they exist
            numeric_cols_for_rolling = df.select_dtypes(include=['number']).columns.tolist()
            no_rolling_features = { # Exclude targets, greeks, date features, etc.
                *target_cols, *greek_columns, *cyclical_date_features, *original_date_features,
                "inTheMoney", "ticker", "risk_free_rate", "daysToExpiry"
                }
            rolling_base_feature_cols = [ col for col in base_numeric_features
                if col in numeric_cols_for_rolling and col not in no_rolling_features ]

            for window in window_sizes:
                 # ... (rest of rolling window logic remains the same) ...
                 if verbose: logging.debug(f"  Processing window size {window}...")
                 for col in rolling_base_feature_cols:
                     if col in df.columns and df[col].count() >= window:
                         try:
                             rolling_series = df[col].rolling(window, min_periods=max(1, window // 2))
                             rolling_features_data[f'{col}_mean_{window}d'] = rolling_series.mean()
                             rolling_features_data[f'{col}_std_{window}d'] = rolling_series.std()
                         except Exception as roll_err: logging.warning(f"Could not calculate rolling for {col} (w {window}): {roll_err}")
                     elif col in df.columns and verbose: logging.debug(f" Skipping rolling for {col} (w {window}): insufficient data")

            if rolling_features_data:
                rolling_df = pd.DataFrame(rolling_features_data); rolling_df.fillna(0, inplace=True)
                df = pd.concat([df, rolling_df], axis=1)
                if verbose: logging.info(f"Generated {len(rolling_df.columns)} rolling features.")
            elif verbose: logging.info("No rolling features generated.")
        else:
             if verbose: logging.info("Skipping rolling window feature computation.")


        # --- Final Feature Selection ---
        final_feature_cols = [col for col in base_numeric_features if col in df.columns] # Start with base

        # Conditionally add rolling features
        rolling_feature_names = list(rolling_features_data.keys())
        if include_rolling_features:
            final_feature_cols.extend(col for col in rolling_feature_names if col in df.columns)
            if verbose: logging.info(f"Including {len([c for c in rolling_feature_names if c in df.columns])} rolling features.")
        else:
            if verbose: logging.info("Excluding rolling features as requested.")

        # Conditionally add Greek features
        actual_greeks_present = [col for col in greek_columns if col in df.columns]
        if include_greeks:
            if actual_greeks_present:
                final_feature_cols.extend(actual_greeks_present)
                if verbose: logging.info(f"Including {len(actual_greeks_present)} Greek features: {actual_greeks_present}")
            else: logging.warning("Greeks requested but no Greek columns found.")
        else:
            if verbose: logging.info("Excluding Greek features as requested.")

        # --- MODIFIED: Conditionally add Date features ---
        if include_cyclical_features:
            date_features_to_add = [col for col in cyclical_date_features if col in df.columns]
            if date_features_to_add:
                final_feature_cols.extend(date_features_to_add)
                if verbose: logging.info(f"Including {len(date_features_to_add)} cyclical date features.")
            else: logging.warning("Cyclical date features requested but not found.")
        else:
            date_features_to_add = [col for col in original_date_features if col in df.columns]
            if date_features_to_add:
                final_feature_cols.extend(date_features_to_add)
                if verbose: logging.info(f"Including {len(date_features_to_add)} original date features.")
            else: logging.warning("Original date features requested but not found.")
        # --- END MODIFICATION ---

        # --- Target Column Validation & Final Setup ---
        valid_target_cols = [col for col in target_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if len(valid_target_cols) != len(target_cols):
             missing_or_invalid = set(target_cols) - set(valid_target_cols)
             logging.warning(f"Target columns missing/invalid: {missing_or_invalid}. Using: {valid_target_cols}")
        self.target_cols = valid_target_cols
        self.n_targets = len(self.target_cols)
        if self.n_targets == 0: raise ValueError("No valid target columns found.")

        # Use only features that actually exist in the dataframe AFTER potential additions
        self.feature_cols = sorted(list(set(col for col in final_feature_cols if col in df.columns)))
        self.n_features = len(self.feature_cols)
        if self.n_features == 0: raise ValueError("No features available in the dataset.")

        all_needed_cols = self.feature_cols + self.target_cols
        initial_rows = len(df)
        df = df.dropna(subset=all_needed_cols).reset_index(drop=True)
        rows_after_dropna = len(df)
        if verbose and initial_rows > 0: logging.info(f"Dropped {initial_rows - rows_after_dropna} rows ({((initial_rows - rows_after_dropna)/initial_rows)*100:.2f}%) due to NaNs in needed columns.")

        # Convert final data to numpy
        if not df.empty:
             try: self.data = df[all_needed_cols].to_numpy(dtype=np.float32)
             except Exception as e: logging.error(f"Error converting final DataFrame to NumPy: {e}"); raise
        else: self.data = np.empty((0, self.n_features + self.n_targets), dtype=np.float32)

        self.seq_len = seq_len; self.n_samples = self.data.shape[0]
        self.max_index = max(-1, self.n_samples - self.seq_len -1); self.ticker = ticker

        if verbose or self.n_samples < self.seq_len + 1:
            logging.info(f"Dataset created for {ticker}: Features={self.n_features}, Targets={self.n_targets}, Samples={self.n_samples}, MaxIdx={self.max_index}")
            if self.n_samples < self.seq_len + 1: logging.warning(f"Insufficient samples ({self.n_samples}) for sequence length ({self.seq_len}).")
            logging.info(f"Final features used ({self.n_features}): {self.feature_cols}")

    def _set_empty_attributes(self, target_cols, seq_len, ticker):
        """Helper to set attributes for an empty dataset."""
        self.data = np.empty((0, 0), dtype=np.float32)
        self.n_features = 0
        self.n_targets = len(target_cols)
        self.seq_len = seq_len
        self.n_samples = 0
        self.max_index = -1
        self.feature_cols = []
        self.target_cols = target_cols
        self.ticker = ticker

    def __len__(self):
        return max(0, self.max_index + 1)

    def __getitem__(self, idx):
        if idx > self.max_index:
            raise IndexError(f"Index {idx} out of bounds (max index: {self.max_index})")
        start_idx = idx; end_idx = idx + self.seq_len; target_idx = idx + self.seq_len
        if end_idx > self.data.shape[0] or target_idx >= self.data.shape[0]:
             raise IndexError(f"Calculated indices exceed data bounds for input index {idx}.")

        x_seq = self.data[start_idx : end_idx, :self.n_features]
        y_val = self.data[target_idx, self.n_features : self.n_features + self.n_targets]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)