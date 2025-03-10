import pandas as pd
import numpy as np

def encode_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Converts 'lastTradeDate' to datetime (if not already).
    2. Creates day-of-week (Monday=0), day-of-month, and day-of-year columns.
    3. Applies sine/cosine transforms for cyclical encoding of each.
    4. Preserves the original columns for optional future use.
    """
    # Ensure 'lastTradeDate' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['lastTradeDate']):
        df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'])
    
    # Day of week (Monday=0, Sunday=6)
    df['day_of_week'] = df['lastTradeDate'].dt.dayofweek
    
    # Day of month (1..31)
    df['day_of_month'] = df['lastTradeDate'].dt.day
    
    # Day of year (1..365 or 366)
    df['day_of_year'] = df['lastTradeDate'].dt.dayofyear
    
    # -------------------------
    # Cyclical encoding
    # day_of_week => 0..6
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # day_of_month => 1..31
    # Use (day_of_month - 1) so it starts at 0..30 for the transform
    df['day_of_month_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31)
    
    # day_of_year => 1..365 (or 366). We'll assume 365 for simplicity.
    df['day_of_year_sin'] = np.sin(2 * np.pi * (df['day_of_year'] - 1) / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)
    
    return df


def main():
    data_path = '/Users/bekheet/dev/option-ml-prediction/data_files/option_data.csv'
    df = pd.read_csv(data_path)
    
    # Encode the time features
    df_encoded = encode_temporal_features(df)
    
    # Display a few rows to confirm the new columns
    print(df_encoded.head())
    
    # (Optional) Save out to a new file
    output_path = '/Users/bekheet/dev/option-ml-prediction/data_files/option_data_time_encoded.csv'
    df_encoded.to_csv(output_path, index=False)
    print(f"Time-encoded data saved to: {output_path}")


if __name__ == "__main__":
    main()
