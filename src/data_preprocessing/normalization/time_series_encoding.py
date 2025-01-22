import pandas as pd
import numpy as np
from tqdm import tqdm

def encode_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Converts 'lastTradeDate' to datetime (if not already).
    2. Creates day-of-week (Monday=0), day-of-month, and day-of-year columns.
    3. Applies sine/cosine transforms for cyclical encoding of each.
    4. Preserves the original columns for optional future use.
    """
    print("Encoding temporal features...")
    
    # Ensure 'lastTradeDate' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['lastTradeDate']):
        print("Converting lastTradeDate to datetime...")
        tqdm.pandas(desc="Converting dates")
        df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'], progress_bar=True)

    # Create a progress bar for the feature creation steps
    steps = ['day_of_week', 'day_of_month', 'day_of_year', 
            'cyclical_week', 'cyclical_month', 'cyclical_year']
    pbar = tqdm(steps, desc="Creating features")

    # Day of week (Monday=0, Sunday=6)
    df['day_of_week'] = df['lastTradeDate'].dt.dayofweek
    pbar.update(1)

    # Day of month (1..31)
    df['day_of_month'] = df['lastTradeDate'].dt.day
    pbar.update(1)

    # Day of year (1..365 or 366)
    df['day_of_year'] = df['lastTradeDate'].dt.dayofyear
    pbar.update(1)

    # -------------------------
    # Cyclical encoding
    # day_of_week => 0..6
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    pbar.update(1)

    # day_of_month => 1..31
    # Use (day_of_month - 1) so it starts at 0..30 for the transform
    df['day_of_month_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31)
    pbar.update(1)

    # day_of_year => 1..365 (or 366). We'll assume 365 for simplicity.
    df['day_of_year_sin'] = np.sin(2 * np.pi * (df['day_of_year'] - 1) / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)
    pbar.update(1)
    
    pbar.close()
    return df

def main():
    print("Starting data processing...")
    
    # Show progress bar for reading CSV
    data_path = '/Users/bekheet/dev/option-ml-prediction/data_files/option_data.csv'
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    # Encode the time features
    df_encoded = encode_temporal_features(df)

    # Display a few rows to confirm the new columns
    print("\nFirst few rows of encoded data:")
    print(df_encoded.head())

    # Save with progress bar
    output_path = '/Users/bekheet/dev/option-ml-prediction/data_files/option_data_time_encoded.csv'
    print(f"\nSaving encoded data to: {output_path}")
    
    # Using tqdm to wrap the to_csv operation
    with tqdm(total=1, desc="Saving CSV") as pbar:
        df_encoded.to_csv(output_path, index=False)
        pbar.update(1)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()