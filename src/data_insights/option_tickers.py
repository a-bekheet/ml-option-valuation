import pandas as pd
import numpy as np

def analyze_tickers():
    """
    Extract and analyze tickers from the options data
    """
    # Read the data
    print("Reading options data...")
    df = pd.read_csv('data_files/option_data_no_BACC.csv')
    
    # Get unique tickers
    tickers = df['ticker'].unique()
    
    # Create ticker analysis
    print(f"\nFound {len(tickers)} unique tickers:")
    print("=" * 50)
    
    # Analyze each ticker
    ticker_analysis = []
    for ticker in sorted(tickers):  # Sort alphabetically
        ticker_data = df[df['ticker'] == ticker]
        
        # Get date range
        date_range = pd.to_datetime(ticker_data['quoteDate'])
        start_date = date_range.min()
        end_date = date_range.max()
        
        # Get price range
        price_range = ticker_data['stockClose']
        min_price = price_range.min()
        max_price = price_range.max()
        
        # Collect analysis
        analysis = {
            'ticker': ticker,
            'data_points': len(ticker_data),
            'start_date': start_date,
            'end_date': end_date,
            'min_price': min_price,
            'max_price': max_price,
            'unique_strikes': ticker_data['strike'].nunique(),
            'total_volume': ticker_data['volume'].sum()
        }
        ticker_analysis.append(analysis)
    
    # Convert to DataFrame for better display
    analysis_df = pd.DataFrame(ticker_analysis)
    
    # Print summary
    print("\nTicker Analysis Summary:")
    for _, row in analysis_df.iterrows():
        print(f"\n{row['ticker']}:")
        print(f"  Date Range: {row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}")
        print(f"  Price Range: ${row['min_price']:.2f} to ${row['max_price']:.2f}")
        print(f"  Data Points: {row['data_points']:,}")
        print(f"  Unique Strikes: {row['unique_strikes']:,}")
        print(f"  Total Volume: {row['total_volume']:,}")
    
    # Save analysis
    analysis_df.to_csv('data_insights/ticker_analysis.csv', index=False)
    print(f"\nSaved detailed analysis to ticker_analysis.csv")
    
    # Return tickers list
    return sorted(tickers)

if __name__ == "__main__":
    tickers = analyze_tickers()
    
    # Print tickers in a format ready for use
    print("\nTickers list for reference:")
    print("['" + "', '".join(tickers) + "']")