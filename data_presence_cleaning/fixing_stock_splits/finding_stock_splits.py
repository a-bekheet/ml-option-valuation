import pandas as pd
import numpy as np
import logging
from typing import List

def detect_stock_splits(df: pd.DataFrame, split_ratios: List[int] = [2, 3, 4, 5], min_rows: int = 50) -> pd.DataFrame:
    """
    Detect potential stock splits by comparing consecutive day prices

    Args:
        df: DataFrame with stock data
        split_ratios: List of split ratios to detect (e.g., [2, 3, 4] for 2:1, 3:1, 4:1 splits)
        min_rows: Minimum number of rows required for a ticker to be considered for split detection
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Extract ticker from contractSymbol
    df['ticker'] = df['contractSymbol'].str.extract(r'([A-Za-z]+)')[0]

    # Group by ticker and sort by date
    df = df.sort_values('quoteDate')
    grouped = df.groupby('ticker')

    # Filter out tickers with insufficient data
    tickers_to_analyze = grouped.filter(lambda x: len(x) >= min_rows)['ticker'].unique()

    potential_splits = []

    for ticker in tickers_to_analyze:
        ticker_data = grouped.get_group(ticker)

        # Calculate daily price ratios
        price_ratios = ticker_data['stockClose'] / ticker_data['stockClose'].shift(1)

        for ratio in split_ratios:
            split_threshold_low = 1 / (ratio + 0.1)
            split_threshold_high = 1 / (ratio - 0.1)

            # Detect potential splits
            split_indices = np.where((price_ratios < split_threshold_low) | (price_ratios > split_threshold_high))[0]

            if len(split_indices) > 0:
                split_dates = ticker_data.iloc[split_indices]['quoteDate'].tolist()
                split_ratios_detected = price_ratios.iloc[split_indices].tolist()

                for date, detected_ratio in zip(split_dates, split_ratios_detected):
                    potential_splits.append({
                        'ticker': ticker,
                        'split_date': date,
                        'split_ratio': detected_ratio
                    })

    potential_splits_df = pd.DataFrame(potential_splits)

    # Log summary statistics
    logging.info(f"Total potential splits detected: {len(potential_splits_df)}")
    logging.info(f"Unique tickers with potential splits: {potential_splits_df['ticker'].nunique()}")
    logging.info(f"Date range: {potential_splits_df['split_date'].min()} to {potential_splits_df['split_date'].max()}")

    return potential_splits_df

def adjust_for_splits(df: pd.DataFrame, potential_splits_df: pd.DataFrame, price_columns: List[str] = ['strike', 'lastPrice', 'bid', 'ask', 'stockClose', 'stockHigh', 'stockLow', 'stockOpen']) -> pd.DataFrame:
    """
    Adjust option prices and strikes for potential stock splits

    Args:
        df: DataFrame with option data
        potential_splits_df: DataFrame with potential stock splits
        price_columns: List of columns to adjust for splits
    """
    adjusted_df = df.copy()

    for _, split_row in potential_splits_df.iterrows():
        ticker = split_row['ticker']
        split_date = pd.to_datetime(split_row['split_date'])
        split_ratio = split_row['split_ratio']

        # Adjust prices and strikes
        ticker_mask = (adjusted_df['ticker'] == ticker) & (pd.to_datetime(adjusted_df['quoteDate']) < split_date)
        adjusted_df.loc[ticker_mask, price_columns] /= split_ratio

    return adjusted_df

def analyze_price_consistency(df: pd.DataFrame) -> None:
    """
    Analyze price consistency and relationships
    """
    logging.info("Analyzing price consistency...")

    # Check bid-ask relationship
    invalid_spreads = df[df['bid'] > df['ask']]
    logging.info(f"Invalid bid-ask spreads found: {len(invalid_spreads)}")

    # Check strike vs stock price for ITM options
    calls = df[df['contractSymbol'].str.contains('C')]
    puts = df[df['contractSymbol'].str.contains('P')]

    itm_calls = calls[calls['strike'] < calls['stockClose']]
    itm_puts = puts[puts['strike'] > puts['stockClose']]

    logging.info(f"ITM Calls analysis:")
    logging.info(f"Total ITM calls: {len(itm_calls)}")
    logging.info(f"Average ITM amount: ${(itm_calls['stockClose'] - itm_calls['strike']).mean():.2f}")

    logging.info(f"ITM Puts analysis:")
    logging.info(f"Total ITM puts: {len(itm_puts)}")
    logging.info(f"Average ITM amount: ${(itm_puts['strike'] - itm_puts['stockClose']).mean():.2f}")

def main() -> None:
    # Read the data
    df = pd.read_csv('data_files/option_data_no_BACC.csv', parse_dates=['quoteDate'])

    # Detect potential splits
    potential_splits_df = detect_stock_splits(df)

    # List unique tickers with potential splits and their dates
    unique_tickers = potential_splits_df.groupby('ticker')['split_date'].apply(list).reset_index()
    logging.info("Unique Tickers with Potential Splits:")
    for _, row in unique_tickers.iterrows():
        ticker = row['ticker']
        split_dates = row['split_date']
        logging.info(f"Ticker: {ticker}, Split Dates: {split_dates}")

    # Adjust for splits
    df_adjusted = adjust_for_splits(df, potential_splits_df)

    # Analyze price consistency
    analyze_price_consistency(df_adjusted)

    # Save adjusted data
    df_adjusted.to_csv('data_files/option_data_split_adjusted.csv', index=False)

if __name__ == "__main__":
    main()
