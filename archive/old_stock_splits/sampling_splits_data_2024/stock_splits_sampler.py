import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_split_data(file_path: str):
    """
    Check data availability around known split dates.
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Convert date columns
    date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].str.split('+').str[0])
    
    # Known splits to check
    splits_to_check = {
        'TSLA': pd.to_datetime('2022-08-25'),
        'GOOGL': pd.to_datetime('2022-07-18')
    }
    
    for ticker, split_date in splits_to_check.items():
        print(f"\nChecking {ticker} split on {split_date.date()}:")
        
        # Get data for this ticker
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if len(ticker_data) == 0:
            print(f"No data found for {ticker}")
            continue
        
        # Get date range of data
        date_range = ticker_data['quoteDate'].agg(['min', 'max'])
        print(f"Data range: {date_range['min'].date()} to {date_range['max'].date()}")
        
        # Check 5 days before and after split
        before_split = split_date - timedelta(days=5)
        after_split = split_date + timedelta(days=5)
        
        around_split = ticker_data[
            (ticker_data['quoteDate'] >= before_split) & 
            (ticker_data['quoteDate'] <= after_split)
        ]
        
        if len(around_split) == 0:
            print("No data found around split date!")
            continue
        
        print("\nData around split date:")
        print("------------------------")
        
        # Group by date and get key metrics
        daily_data = around_split.groupby('quoteDate').agg({
            'stockClose': 'mean',
            'stockAdjClose': 'mean',
            'strike': ['count', 'mean'],
            'volume': 'sum'
        }).round(2)
        
        # Format the output
        daily_data.columns = ['stockClose', 'stockAdjClose', 'num_options', 'avg_strike', 'volume']
        print(daily_data.to_string())
        
        # Calculate key ratios
        print("\nKey metrics:")
        print("------------")
        
        # Get prices just before and after split
        before_price = ticker_data[ticker_data['quoteDate'] < split_date]['stockClose'].mean()
        after_price = ticker_data[ticker_data['quoteDate'] > split_date]['stockClose'].mean()
        
        if before_price and after_price:
            ratio = before_price / after_price
            print(f"Price ratio (before/after): {ratio:.2f}")
            
        # Check strike price changes
        before_strikes = ticker_data[ticker_data['quoteDate'] < split_date]['strike'].mean()
        after_strikes = ticker_data[ticker_data['quoteDate'] > split_date]['strike'].mean()
        
        if before_strikes and after_strikes:
            strike_ratio = before_strikes / after_strikes
            print(f"Strike ratio (before/after): {strike_ratio:.2f}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    check_split_data('data_files/option_data_no_BACC.csv')