import pandas as pd

def detect_stock_splits(df, threshold=0.45):
    """
    Detect potential stock splits by comparing consecutive day prices
    
    Args:
        df: DataFrame with stock data
        threshold: Maximum ratio to consider as split (e.g., 0.45 for detecting 2:1 splits)
    """
    # Group by ticker (extract from contractSymbol) and sort by date
    def extract_ticker(symbol):
        # Extract ticker from contractSymbol (e.g., 'AAPL220218C00150000' -> 'AAPL')
        return ''.join(filter(str.isalpha, symbol.split('220')[0]))
    
    df['ticker'] = df['contractSymbol'].apply(extract_ticker)
    
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'quoteDate'])
    
    # Calculate daily price ratios
    df['price_ratio'] = df.groupby('ticker')['stockClose'].transform(lambda x: x / x.shift(1))
    
    # Detect potential splits
    potential_splits = df[
        (df['price_ratio'] < threshold) | 
        (df['price_ratio'] > (1/threshold))
    ].copy()
    
    # Summarize potential splits
    summary = potential_splits.groupby('ticker').agg(
        split_dates=('quoteDate', list),
        split_ratios=('price_ratio', list)
    ).reset_index()
    
    print("\nPotential Stock Splits Detected:")
    for _, row in summary.iterrows():
        print(f"\nTicker: {row['ticker']}")
        for date, ratio in zip(row['split_dates'], row['split_ratios']):
            print(f"Date: {date}, Ratio: {ratio:.4f}")
    
    return potential_splits

def adjust_for_splits(df, known_splits):
    """
    Adjust option prices and strikes for known stock splits
    
    Args:
        df: DataFrame with option data
        known_splits: Dictionary of known splits
        Example: {
            'AAPL': [{'date': '2020-08-31', 'ratio': 4}],
            'TSLA': [{'date': '2020-08-31', 'ratio': 5}]
        }
    """
    df = df.copy()
    
    def adjust_prices(row):
        ticker = ''.join(filter(str.isalpha, row['contractSymbol'].split('220')[0]))
        if ticker in known_splits:
            for split in known_splits[ticker]:
                split_date = pd.to_datetime(split['date'])
                if pd.to_datetime(row['quoteDate']) < split_date:
                    # Adjust prices and strikes
                    row['strike'] = row['strike'] / split['ratio']
                    row['lastPrice'] = row['lastPrice'] / split['ratio']
                    row['bid'] = row['bid'] / split['ratio']
                    row['ask'] = row['ask'] / split['ratio']
                    row['stockClose'] = row['stockClose'] / split['ratio']
                    row['stockHigh'] = row['stockHigh'] / split['ratio']
                    row['stockLow'] = row['stockLow'] / split['ratio']
                    row['stockOpen'] = row['stockOpen'] / split['ratio']
        return row
    
    df = df.apply(adjust_prices, axis=1)
    return df

def analyze_price_consistency(df):
    """
    Analyze price consistency and relationships
    """
    print("\nPrice Consistency Analysis:")
    print("-" * 50)
    
    # Check bid-ask relationship
    invalid_spreads = df[df['bid'] > df['ask']]
    print(f"\nInvalid bid-ask spreads found: {len(invalid_spreads)}")
    
    # Check strike vs stock price for ITM options
    calls = df[df['contractSymbol'].str.contains('C')]
    puts = df[df['contractSymbol'].str.contains('P')]
    
    itm_calls = calls[calls['strike'] < calls['stockClose']]
    itm_puts = puts[puts['strike'] > puts['stockClose']]
    
    print(f"\nITM Calls analysis:")
    print(f"Total ITM calls: {len(itm_calls)}")
    print(f"Average ITM amount: ${(itm_calls['stockClose'] - itm_calls['strike']).mean():.2f}")
    
    print(f"\nITM Puts analysis:")
    print(f"Total ITM puts: {len(itm_puts)}")
    print(f"Average ITM amount: ${(itm_puts['strike'] - itm_puts['stockClose']).mean():.2f}")

def main():
    # Read the data
    df = pd.read_csv('data_files/option_data_compressed.csv')
    
    # Detect potential splits
    potential_splits = detect_stock_splits(df)
    print(potential_splits.sample)
    print(potential_splits.summary.describe)

    # known_splits = {
    #     # Example:
    #     # 'AAPL': [{'date': '2020-08-31', 'ratio': 4}],
    #     # 'TSLA': [{'date': '2020-08-31', 'ratio': 5}]
    # }
    
    # # Adjust for splits
    # df_adjusted = adjust_for_splits(df, known_splits)
    
    # # Analyze price consistency
    # analyze_price_consistency(df_adjusted)
    
    # Save adjusted data
    # df_adjusted.to_csv('data_files/option_data_split_adjusted.csv', index=False)

if __name__ == "__main__":
    main()