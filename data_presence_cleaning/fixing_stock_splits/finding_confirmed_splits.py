import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import concurrent.futures
import time

def extract_ticker_info(df):
    """
    Extract unique tickers and their date ranges from contract symbols
    """
    def extract_ticker(symbol):
        # Extract ticker from contractSymbol (e.g., 'AAPL220218C00150000' -> 'AAPL')
        return ''.join(filter(str.isalpha, symbol.split('220')[0]))
    
    # Add ticker column
    df['ticker'] = df['contractSymbol'].apply(extract_ticker)
    
    # Get date range for each ticker
    ticker_info = {}
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]
        start_date = pd.to_datetime(ticker_data['quoteDate']).min()
        end_date = pd.to_datetime(ticker_data['quoteDate']).max()
        
        ticker_info[ticker] = {
            'start_date': start_date,
            'end_date': end_date,
            'data_points': len(ticker_data)
        }
    
    return ticker_info

def get_stock_splits(ticker, start_date, end_date, retry_count=3):
    """
    Get stock splits for a ticker using yfinance with retry logic
    """
    for attempt in range(retry_count):
        try:
            # Add buffer to dates to catch splits near the boundaries
            buffer_days = 30
            start_with_buffer = start_date - timedelta(days=buffer_days)
            end_with_buffer = end_date + timedelta(days=buffer_days)
            
            # Get stock data
            stock = yf.Ticker(ticker)
            splits = stock.splits
            
            # Filter splits within our date range (including buffer)
            relevant_splits = splits[
                (splits.index >= start_with_buffer) & 
                (splits.index <= end_with_buffer)
            ]
            
            if not relevant_splits.empty:
                return {
                    'ticker': ticker,
                    'splits': [
                        {
                            'date': date.strftime('%Y-%m-%d'),
                            'ratio': ratio,
                        }
                        for date, ratio in relevant_splits.items()
                    ]
                }
            return None
            
        except Exception as e:
            if attempt == retry_count - 1:  # Last attempt
                print(f"Failed to get splits for {ticker} after {retry_count} attempts: {str(e)}")
                return None
            time.sleep(1)  # Wait before retrying

def process_splits_data(filepath='data_files/option_data_compressed.csv'):
    """
    Process options data to find and record all stock splits
    """
    print("Reading options data...")
    df = pd.read_csv(filepath)
    
    # Get ticker info
    print("\nExtracting ticker information...")
    ticker_info = extract_ticker_info(df)
    
    print(f"\nFound {len(ticker_info)} unique tickers")
    
    # Get splits data using parallel processing
    print("\nFetching stock splits data...")
    splits_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(
                get_stock_splits,
                ticker,
                info['start_date'],
                info['end_date']
            ): ticker
            for ticker, info in ticker_info.items()
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            print(f"Progress: {completed}/{len(ticker_info)} tickers processed", end='\r')
            
            try:
                splits = future.result()
                if splits and splits['splits']:
                    splits_data.append(splits)
            except Exception as e:
                print(f"\nError processing {ticker}: {str(e)}")
    
    # Create summary report
    print("\n\nGenerating splits summary...")
    
    if splits_data:
        print("\nStock Splits Found:")
        print("=" * 50)
        for split_info in splits_data:
            ticker = split_info['ticker']
            print(f"\n{ticker}:")
            for split in split_info['splits']:
                print(f"  Date: {split['date']}")
                print(f"  Ratio: {split['ratio']}")
                print(f"  Data range: {ticker_info[ticker]['start_date'].strftime('%Y-%m-%d')} to "
                      f"{ticker_info[ticker]['end_date'].strftime('%Y-%m-%d')}")
        
        # Save splits data
        splits_df = pd.DataFrame([
            {
                'ticker': split_info['ticker'],
                'split_date': split['date'],
                'split_ratio': split['ratio']
            }
            for split_info in splits_data
            for split in split_info['splits']
        ])
        
        splits_df.to_csv('stock_splits.csv', index=False)
        print(f"\nSaved splits data to stock_splits.csv")
        
        # Return splits in format ready for adjustment
        return {
            split_info['ticker']: [
                {
                    'date': split['date'],
                    'ratio': split['ratio']
                }
                for split in split_info['splits']
            ]
            for split_info in splits_data
        }
    else:
        print("\nNo stock splits found in the data range")
        return {}

def verify_splits_impact(df, splits_dict):
    """
    Verify the impact of detected splits on the data
    """
    print("\nVerifying splits impact...")
    
    for ticker, splits in splits_dict.items():
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data['quoteDate'] = pd.to_datetime(ticker_data['quoteDate'])
        
        for split in splits:
            split_date = pd.to_datetime(split['date'])
            
            # Get prices around split date
            before_split = ticker_data[
                ticker_data['quoteDate'] < split_date
            ]['stockClose'].mean()
            
            after_split = ticker_data[
                ticker_data['quoteDate'] >= split_date
            ]['stockClose'].mean()
            
            if before_split and after_split:
                actual_ratio = before_split / after_split
                expected_ratio = split['ratio']
                
                print(f"\n{ticker} Split on {split['date']}:")
                print(f"Expected ratio: {expected_ratio:.2f}")
                print(f"Actual ratio: {actual_ratio:.2f}")
                print(f"Average price before: ${before_split:.2f}")
                print(f"Average price after: ${after_split:.2f}")

if __name__ == "__main__":
    # Process splits
    splits_dict = process_splits_data()
    
    if splits_dict:
        # Read data again for verification
        df = pd.read_csv('data_files/option_data_compressed.csv')
        
        # Add ticker column for verification
        df['ticker'] = df['contractSymbol'].apply(
            lambda x: ''.join(filter(str.isalpha, x.split('220')[0]))
        )
        
        # Verify splits impact
        verify_splits_impact(df, splits_dict)
    
    print("\nProcessing complete!")