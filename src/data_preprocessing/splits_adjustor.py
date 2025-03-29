import logging
import re
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def extract_ticker_from_symbol(symbol_series: pd.Series) -> pd.Series:
    """
    Extracts ticker symbol from option contractSymbol series.
    Handles potential errors gracefully.
    """
    def _extract(symbol):
        # Extract ticker from contractSymbol (e.g., 'AAPL220218C00150000' -> 'AAPL')
        # Make sure symbol is a string before applying regex
        if pd.isna(symbol):
            return None
        match = re.match(r'^([A-Z]+)', str(symbol))
        return match.group(1) if match else None

    return symbol_series.apply(_extract)

def adjust_for_stock_splits(
    data_path: str,
    output_path: Optional[str] = None,
    return_dataframe: bool = False # Changed default to False as pipeline uses files
) -> Optional[pd.DataFrame]:
    """
    Adjust stock and option data for historical stock splits.
    Now includes ticker extraction if 'ticker' column is missing.

    Args:
        data_path: Path to input CSV file (expects 'contractSymbol' if 'ticker' missing).
        output_path: Path to output CSV file (default: input filename with '_split_adjusted' suffix).
        return_dataframe: Whether to return the adjusted DataFrame.

    Returns:
        Adjusted DataFrame if return_dataframe is True, otherwise None.
    """
    logging.info(f"Starting stock split adjustment for: {data_path}")
    # Set output path if not provided
    if output_path is None:
        output_path = data_path.replace('.csv', '_split_adjusted.csv')

    # Manually verified splits with exact ratios
    verified_splits = {
        'AMZN': {'date': '2022-06-06', 'ratio': 20.0, 'type': 'forward'},
        'GOOGL': {'date': '2022-07-18', 'ratio': 20.0, 'type': 'forward'},
        'TSLA': {'date': '2022-08-25', 'ratio': 3.0, 'type': 'forward'},
        'SHOP': {'date': '2022-07-04', 'ratio': 10.0, 'type': 'forward'},
        'CGC': {'date': '2023-12-20', 'ratio': 10.0, 'type': 'reverse'} # 1:10 reverse
    }

    # Features to adjust
    price_features = [
        'strike', 'lastPrice', 'bid', 'ask', 'change',
        'stockClose', 'stockOpen', 'stockHigh', 'stockLow', 'stockAdjClose', # Added stockAdjClose
        'strikeDelta', 'stockClose_ewm_5d', 'stockClose_ewm_15d',
        'stockClose_ewm_45d', 'stockClose_ewm_135d'
    ]
    volume_features = ['volume', 'openInterest', 'stockVolume']

    try:
        # Load data
        logging.info("Loading data for split adjustment...")
        df = pd.read_csv(data_path)
        logging.info(f"Loaded {len(df)} rows.")

        # --- NEW: Check for 'ticker' column and extract if missing ---
        if 'ticker' not in df.columns:
            logging.warning("'ticker' column not found. Attempting to extract from 'contractSymbol'.")
            if 'contractSymbol' in df.columns:
                df['ticker'] = extract_ticker_from_symbol(df['contractSymbol'])
                missing_tickers = df['ticker'].isna().sum()
                if missing_tickers > 0:
                    logging.warning(f"Could not extract ticker for {missing_tickers} rows from 'contractSymbol'. These rows may not be adjusted.")
                df.dropna(subset=['ticker'], inplace=True) # Drop rows where ticker couldn't be extracted
                logging.info("Added 'ticker' column based on 'contractSymbol'.")
            else:
                logging.error("'ticker' and 'contractSymbol' columns both missing. Cannot perform split adjustments.")
                # Save unchanged data and exit or raise error
                df.to_csv(output_path, index=False)
                return df if return_dataframe else None
        # --- END NEW ---


        # Convert date columns (ensure this runs AFTER loading)
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                 try:
                      df[col] = pd.to_datetime(df[col], errors='coerce')
                 except Exception as e:
                      logging.warning(f"Could not convert column '{col}' to datetime: {e}")


        # Create copy for adjustment
        df_adjusted = df.copy()

        # Convert numeric columns to float64 for adjustment precision
        numeric_features_to_adjust = price_features + volume_features
        for col in numeric_features_to_adjust:
            if col in df_adjusted.columns:
                df_adjusted[col] = pd.to_numeric(df_adjusted[col], errors='coerce').astype('float64')

        # Process each split
        logging.info("Processing defined stock splits...")
        total_adjusted_records = 0
        for ticker, split_info in verified_splits.items():
            logging.debug(f"Processing split for {ticker} on {split_info['date']}")
            split_date = pd.to_datetime(split_info['date'])
            split_ratio = split_info['ratio']
            split_type = split_info['type']

            # Get pre-split data mask using the 'ticker' column (now guaranteed to exist or skipped)
            # Also ensure quoteDate is valid datetime for comparison
            if 'quoteDate' not in df_adjusted.columns or df_adjusted['quoteDate'].isna().all():
                 logging.warning(f"Skipping split adjustment for {ticker}: 'quoteDate' column missing or empty.")
                 continue

            ticker_mask = df_adjusted['ticker'] == ticker
            # Ensure date comparison works by handling potential NaT dates
            pre_split_mask = ticker_mask & df_adjusted['quoteDate'].notna() & (df_adjusted['quoteDate'] < split_date)
            affected_records = pre_split_mask.sum()

            if affected_records == 0:
                logging.debug(f"No pre-split records found for {ticker} before {split_info['date']}. Skipping.")
                continue

            logging.info(f"Adjusting {affected_records} records for {ticker} split ({split_type}, ratio {split_ratio})")
            total_adjusted_records += affected_records

            # Apply adjustments based on split type
            adjustment_ratio = split_ratio if split_type == 'forward' else (1.0 / split_ratio)

            # Adjust price features (Divide by ratio for forward, Multiply for reverse)
            for feature in price_features:
                if feature in df_adjusted.columns:
                    mask = pre_split_mask & df_adjusted[feature].notna()
                    df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] / adjustment_ratio

            # Adjust volume features (Multiply by ratio for forward, Divide for reverse)
            for feature in volume_features:
                if feature in df_adjusted.columns:
                    mask = pre_split_mask & df_adjusted[feature].notna()
                    # Avoid division by zero if adjustment_ratio is somehow 0 (though unlikely)
                    if adjustment_ratio != 0:
                        df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] * adjustment_ratio
                    else:
                         logging.warning(f"Skipping volume adjustment for {feature} due to zero adjustment ratio.")


        logging.info(f"Total records adjusted across all splits: {total_adjusted_records}")
        # Save adjusted data
        logging.info(f"Saving split-adjusted data to: {output_path}")
        df_adjusted.to_csv(output_path, index=False)
        logging.info("Split adjustment step complete.")

        # Return dataframe if requested
        if return_dataframe:
            return df_adjusted
        return None

    except Exception as e:
         logging.error(f"Error during stock split adjustment: {e}")
         logging.error(traceback.format_exc()) # Make sure traceback is imported
         raise