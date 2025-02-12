import pandas as pd
import os
from pathlib import Path

def split_data_by_ticker(input_file, output_dir):
    """
    Split a CSV file containing multiple tickers into separate files by ticker.
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory to save the split files
    
    Returns:
        dict: Mapping of tickers to their file paths and data counts
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get unique tickers and their counts
    ticker_counts = df['ticker'].value_counts().sort_index()
    print(f"\nFound {len(ticker_counts)} unique tickers")
    
    ticker_files = {}
    
    # Process each ticker
    for ticker in ticker_counts.index:
        ticker_data = df[df['ticker'] == ticker].copy()
        output_file = os.path.join(output_dir, f"option_data_scaled_{ticker}.csv")
        
        print(f"Writing {ticker_counts[ticker]:,} rows for {ticker} to {output_file}")
        ticker_data.to_csv(output_file, index=False)
        
        ticker_files[ticker] = {
            'file_path': output_file,
            'count': ticker_counts[ticker]
        }
    
    # Save ticker metadata
    metadata = pd.DataFrame([
        {
            'ticker': ticker,
            'file_path': info['file_path'],
            'count': info['count']
        }
        for ticker, info in ticker_files.items()
    ])
    
    metadata_file = os.path.join(output_dir, 'ticker_metadata.csv')
    metadata.to_csv(metadata_file, index=False)
    print(f"\nSaved ticker metadata to {metadata_file}")
    
    return ticker_files

if __name__ == "__main__":
    # Parse command line arguments or use input
    print("Data Splitting Tool")
    print("-" * 50)
    
    input_file = input("Enter path to input CSV file: ")
    output_dir = input("Enter output directory path: ")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        exit(1)
        
    try:
        ticker_files = split_data_by_ticker(input_file, output_dir)
        
        print("\nSplit complete! Summary:")
        print("-" * 50)
        for ticker, info in ticker_files.items():
            print(f"{ticker}: {info['count']:,} rows -> {os.path.basename(info['file_path'])}")
            
    except Exception as e:
        print(f"Error occurred while splitting data: {str(e)}")
        exit(1)