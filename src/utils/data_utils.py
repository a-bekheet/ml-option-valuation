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
                 include_greeks: bool = True):
        """
        Loads data for a given ticker, adds rolling window features, and optionally includes Greeks.
        Looks for data in the 'by_ticker' subdirectory of data_dir.

        Args:
            data_dir (str): Base directory containing the 'by_ticker' subdirectory
                            (e.g., 'data_files/split_data').
            ticker (str): The stock ticker symbol.
            seq_len (int): Length of the input sequence.
            target_cols (list): List of target column names.
            window_sizes (list): List of window sizes for rolling features.
            verbose (bool): If True, print detailed loading information.
            include_greeks (bool): If True, include calculated Greek columns as features.
        """
        # --- CORRECTED FILE PATH CONSTRUCTION ---
        # Construct the expected file path *within the 'by_ticker'* subdirectory
        # Assumes normalized files with greeks are saved like '{TICKER}_normalized.csv'
        file_path = os.path.join(data_dir, 'by_ticker', f"{ticker}_normalized.csv")
        # --- END CORRECTION ---

        if verbose:
            logging.info(f"Initializing StockOptionDataset for ticker: {ticker}")
            logging.info(f"Attempting to load data from: {file_path}") # Log the corrected path
            logging.info(f"Sequence length: {seq_len}, Target columns: {target_cols}")
            logging.info(f"Include Greeks: {include_greeks}")

        if not os.path.exists(file_path):
            logging.error(f"No data file found for ticker {ticker} at {file_path}")
            # Also check the base directory provided
            logging.error(f"Checked in base directory: {data_dir}")
            raise FileNotFoundError(f"No data file found for ticker {ticker} at expected path: {file_path}")

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                 logging.warning(f"Data file for ticker {ticker} is empty.")
                 self.data = np.empty((0, 0), dtype=np.float32)
                 self.n_features = 0
                 self.n_targets = len(target_cols)
                 self.seq_len = seq_len
                 self.n_samples = 0
                 self.max_index = -1
                 self.feature_cols = []
                 self.target_cols = target_cols
                 self.ticker = ticker
                 return # Stop initializationinitialization

        except pd.errors.EmptyDataError:
             logging.warning(f"Data file for ticker {ticker} is empty or invalid.")
             self.data = np.empty((0, 0), dtype=np.float32)
             self.n_features = 0
             self.n_targets = len(target_cols)
             self.seq_len = seq_len
             self.n_samples = 0
             self.max_index = -1
             self.feature_cols = []
             self.target_cols = target_cols
             self.ticker = ticker
             return
        except Exception as e:
             logging.error(f"Error loading data for ticker {ticker}: {e}")
             raise

        # --- Feature Definition ---
        # Define base features (columns expected BEFORE rolling features and Greeks)
        # Ensure these names match columns present after Step 3.6 in preprocessing
        base_numeric_features = [
            "strike", "lastPrice", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume",
            "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d",
            "day_of_week_sin", "day_of_week_cos", # Cyclical features
            "day_of_month_sin", "day_of_month_cos",
            "day_of_year_sin", "day_of_year_cos",
            "risk_free_rate" # Added rate
            # Note: Removed day_of_week, day_of_month, day_of_year if cyclical are used primarily
        ]

        # Define Greek columns (added during preprocessing)
        greek_columns = ["delta", "gamma", "vega", "theta", "rho"]

        # --- Rolling Window Feature Calculation (Similar to original) ---
        # Identify numeric columns actually present in the loaded dataframe for rolling features
        numeric_cols_for_rolling = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude certain columns from rolling operations (e.g., cyclical, identifiers, maybe greeks?)
        no_rolling_features = { # Add greek_columns here if you DON'T want rolling features based on them
            "day_of_week_sin", "day_of_week_cos", "day_of_month_sin", "day_of_month_cos",
            "day_of_year_sin", "day_of_year_cos", "daysToExpiry", "risk_free_rate",
            "inTheMoney", "ticker" # Add other non-rolling IDs if present
            }.union(set(greek_columns)) # Exclude Greeks from having rolling features calculated *on* them

        rolling_base_feature_cols = [
            col for col in base_numeric_features
            if col in numeric_cols_for_rolling and col not in no_rolling_features
        ]

        rolling_features_data = {}
        if verbose:
            logging.info("Starting rolling window feature computation...")
        for window in window_sizes:
            if verbose:
                logging.debug(f"  Processing window size {window}...")
            for col in rolling_base_feature_cols:
                if df[col].count() > window + 1: # Ensure enough data
                    rolling_series = df[col].rolling(window, min_periods=1) # Use min_periods=1
                    mean_col_name = f'{col}_mean_{window}d'
                    std_col_name = f'{col}_std_{window}d'
                    rolling_features_data[mean_col_name] = rolling_series.mean()
                    rolling_features_data[std_col_name] = rolling_series.std()
                    # Optional: Add rolling pct_change if desired
                    # if not (df[col] == 0).all():
                    #    change_col_name = f'{col}_change_{window}d'
                    #    rolling_features_data[change_col_name] = df[col].pct_change(window) # Needs careful handling of NaNs/Infs
                else:
                     if verbose:
                          logging.debug(f"  Skipping rolling features for '{col}' (window {window}) due to insufficient data.")


        if rolling_features_data:
            rolling_df = pd.DataFrame(rolling_features_data)
            if verbose:
                logging.info(f"Generated {len(rolling_df.columns)} rolling features.")
            # Fill initial NaNs created by rolling std with 0 (or another strategy like backfill)
            rolling_df.fillna(0, inplace=True) # Simple fillna(0) for stddev NaNs at start
            df = pd.concat([df, rolling_df], axis=1)
        elif verbose:
             logging.info("No rolling features generated.")


        # --- Final Feature Selection ---
        # Start with base features that exist in the dataframe
        final_feature_cols = [col for col in base_numeric_features if col in df.columns]

        # Add generated rolling features
        final_feature_cols.extend(rolling_features_data.keys())

        # Conditionally add Greek features
        actual_greeks_present = [col for col in greek_columns if col in df.columns]
        if include_greeks:
            if actual_greeks_present:
                final_feature_cols.extend(actual_greeks_present)
                logging.info(f"Including {len(actual_greeks_present)} Greek features: {actual_greeks_present}")
            else:
                logging.warning("Greeks requested but no Greek columns found in the data file.")
        else:
            logging.info("Excluding Greek features as requested.")

        # Ensure target columns are valid and numeric
        valid_target_cols = [col for col in target_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if len(valid_target_cols) != len(target_cols):
             missing_or_invalid = set(target_cols) - set(valid_target_cols)
             logging.warning(f"Target columns missing or non-numeric: {missing_or_invalid}. Using: {valid_target_cols}")
        self.target_cols = valid_target_cols
        self.n_targets = len(self.target_cols)

        # Ensure final feature columns are unique and exist
        self.feature_cols = sorted(list(set(col for col in final_feature_cols if col in df.columns)))
        self.n_features = len(self.feature_cols)

        # Define combined columns needed for data array, including targets
        all_needed_cols = self.feature_cols + self.target_cols

        # Drop rows with NaN values in ANY of the required feature or target columns
        # This happens AFTER rolling features are calculated and potentially filled
        initial_rows = len(df)
        df = df.dropna(subset=all_needed_cols).reset_index(drop=True)
        rows_after_dropna = len(df)
        if verbose:
             logging.info(f"Dropped {initial_rows - rows_after_dropna} rows due to NaNs in needed columns.")

        # Convert selected data to numpy array
        if not df.empty:
             self.data = df[all_needed_cols].to_numpy(dtype=np.float32)
        else:
             logging.warning(f"DataFrame for {ticker} became empty after dropping NaNs.")
             self.data = np.empty((0, self.n_features + self.n_targets), dtype=np.float32) # Ensure correct shape even if empty


        self.seq_len = seq_len
        self.n_samples = self.data.shape[0]
        # Adjust max_index calculation for sequence length and 0-based indexing
        self.max_index = max(-1, self.n_samples - self.seq_len -1) # Ensure it's at least -1 if no samples

        self.ticker = ticker

        if verbose or self.n_samples < self.seq_len + 1: # Log if data seems insufficient
            logging.info(f"Dataset created for {ticker}:")
            logging.info(f"  Number of features (n_features): {self.n_features}")
            logging.info(f"  Number of targets (n_targets): {self.n_targets}")
            logging.info(f"  Number of samples (rows after dropna): {self.n_samples}")
            logging.info(f"  Sequence length (seq_len): {self.seq_len}")
            logging.info(f"  Max index for __getitem__: {self.max_index}")
            if self.n_samples < self.seq_len + 1:
                 logging.warning(f"  Insufficient samples ({self.n_samples}) to create sequences of length {self.seq_len + 1}.")
            # Log first 5 features for debugging:
            logging.debug(f"  First 5 features used: {self.feature_cols[:5]}")


    def __len__(self):
        # Length is the number of possible start indices for a sequence
        return self.max_index + 1

    def __getitem__(self, idx):
        if idx > self.max_index:
            raise IndexError(f"Index {idx} out of bounds for dataset with max_index {self.max_index}")

        # Features are the first self.n_features columns
        x_seq = self.data[idx : idx + self.seq_len, :self.n_features]
        # Targets are the columns AFTER the features
        y_val = self.data[idx + self.seq_len, self.n_features : self.n_features + self.n_targets]

        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
