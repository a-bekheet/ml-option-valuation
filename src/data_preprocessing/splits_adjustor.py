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
        if pd.isna(symbol): return None
        match = re.match(r'^([A-Z]+)', str(symbol))
        return match.group(1) if match else None
    return symbol_series.apply(_extract)

def adjust_for_stock_splits(
    data_path: str,
    output_path: Optional[str] = None,
    return_dataframe: bool = False
) -> Optional[pd.DataFrame]:
    """
    Adjust stock and option data for historical stock splits.
    Includes ticker extraction and enhanced debugging.

    Args:
        data_path: Path to input CSV file.
        output_path: Path to output CSV file.
        return_dataframe: Whether to return the adjusted DataFrame.

    Returns:
        Adjusted DataFrame if return_dataframe is True, otherwise None.
    """
    logging.info(f"Starting stock split adjustment for: {data_path}")
    if output_path is None:
        output_path = data_path.replace('.csv', '_split_adjusted.csv')

    # Verified splits (same as before)
    verified_splits = {
        'AMZN': {'date': '2022-06-06', 'ratio': 20.0, 'type': 'forward'},
        'GOOGL': {'date': '2022-07-18', 'ratio': 20.0, 'type': 'forward'},
        'TSLA': {'date': '2022-08-25', 'ratio': 3.0, 'type': 'forward'},
        'SHOP': {'date': '2022-07-04', 'ratio': 10.0, 'type': 'forward'},
        'CGC': {'date': '2023-12-20', 'ratio': 10.0, 'type': 'reverse'}
    }
    price_features = [ # Same as before
        'strike', 'lastPrice', 'bid', 'ask', 'change',
        'stockClose', 'stockOpen', 'stockHigh', 'stockLow', 'stockAdjClose',
        'strikeDelta', 'stockClose_ewm_5d', 'stockClose_ewm_15d',
        'stockClose_ewm_45d', 'stockClose_ewm_135d'
    ]
    volume_features = ['volume', 'openInterest', 'stockVolume'] # Same as before

    try:
        logging.info("Loading data for split adjustment...")
        df = pd.read_csv(data_path)
        logging.info(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

        # --- Ensure 'ticker' Column Exists ---
        if 'ticker' not in df.columns:
            logging.warning("'ticker' column not found. Attempting extraction from 'contractSymbol'.")
            if 'contractSymbol' in df.columns:
                df['ticker'] = extract_ticker_from_symbol(df['contractSymbol'])
                missing_tickers = df['ticker'].isna().sum()
                if missing_tickers > 0:
                    logging.warning(f"Could not extract ticker for {missing_tickers} rows. Dropping these rows.")
                    df.dropna(subset=['ticker'], inplace=True) # Drop rows where ticker is NaN after extraction
                if df.empty:
                     logging.error("DataFrame became empty after dropping rows with unextractable tickers.")
                     df.to_csv(output_path, index=False) # Save empty file
                     return df if return_dataframe else None
                logging.info("Added 'ticker' column. Value counts (sample):\n" + df['ticker'].value_counts().head().to_string())
            else:
                logging.error("'ticker' and 'contractSymbol' columns missing. Cannot perform split adjustments.")
                df.to_csv(output_path, index=False)
                return df if return_dataframe else None
        else:
             logging.info("'ticker' column already exists.")
        # --- End Ticker Check ---

        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        df = df.dropna(subset=['quoteDate']) # Drop rows if quoteDate is invalid after conversion

        df_adjusted = df.copy()

        # Convert numeric features to float64
        numeric_features_to_adjust = price_features + volume_features
        for col in numeric_features_to_adjust:
            if col in df_adjusted.columns:
                df_adjusted[col] = pd.to_numeric(df_adjusted[col], errors='coerce').astype('float64')

        logging.info("Processing defined stock splits...")
        total_adjusted_records = 0
        # **Extra Debug**: Check columns *before* the loop
        logging.debug(f"Columns in df_adjusted before split loop: {df_adjusted.columns.tolist()}")
        if 'ticker' not in df_adjusted.columns:
             logging.error("FATAL: 'ticker' column disappeared before split loop!")
             raise KeyError("Ticker column missing unexpectedly before processing splits.")

        for ticker, split_info in verified_splits.items():
            split_date = pd.to_datetime(split_info['date'])
            split_ratio = split_info['ratio']
            split_type = split_info['type']
            logging.debug(f"Processing split for {ticker}...")

            # **Extra Debug**: Confirm 'ticker' column exists right before filtering
            if 'ticker' not in df_adjusted.columns:
                 logging.error(f"FATAL: 'ticker' column missing just before processing {ticker}!")
                 raise KeyError(f"Ticker column missing unexpectedly when processing {ticker}.")

            # Filter pre-split rows for the current ticker
            try:
                 # This is the line that previously caused the error
                 ticker_mask = df_adjusted['ticker'] == ticker
            except KeyError as ke:
                 logging.error(f"Still encountering KeyError for 'ticker' when processing {ticker}!")
                 logging.error(f"Columns available at this point: {df_adjusted.columns.tolist()}")
                 raise ke # Re-raise the error after logging

            pre_split_mask = ticker_mask & df_adjusted['quoteDate'].notna() & (df_adjusted['quoteDate'] < split_date)
            affected_records = pre_split_mask.sum()

            if affected_records == 0: continue

            logging.info(f"Adjusting {affected_records} records for {ticker} split...")
            total_adjusted_records += affected_records
            adjustment_ratio = split_ratio if split_type == 'forward' else (1.0 / split_ratio)

            # Adjust price features
            for feature in price_features:
                if feature in df_adjusted.columns:
                    mask = pre_split_mask & df_adjusted[feature].notna()
                    df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] / adjustment_ratio

            # Adjust volume features
            for feature in volume_features:
                if feature in df_adjusted.columns:
                    mask = pre_split_mask & df_adjusted[feature].notna()
                    if adjustment_ratio != 0:
                        df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] * adjustment_ratio
                    else:
                        logging.warning(f"Skipping vol adjustment for {feature} (zero ratio).")

        logging.info(f"Total records adjusted: {total_adjusted_records}")
        logging.info(f"Saving split-adjusted data to: {output_path}")
        df_adjusted.to_csv(output_path, index=False)
        logging.info("Split adjustment step complete.")

        if return_dataframe: return df_adjusted
        return None

    except Exception as e:
         logging.error(f"Error during stock split adjustment: {e}")
         logging.error(traceback.format_exc())
         raise