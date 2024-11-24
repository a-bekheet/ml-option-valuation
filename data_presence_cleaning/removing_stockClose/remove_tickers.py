import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('data_files/option_data_compressed.csv')

def extract_ticker(symbol):
    # Extract ticker from contractSymbol (e.g., 'AAPL220218C00150000' -> 'AAPL')
    ticker = re.split(r'\d', symbol)[0]
    return ticker

# Add ticker column
df['ticker'] = df['contractSymbol'].apply(extract_ticker)

# Define the list of tickers to drop
tickers_to_drop = ['BAC', 'C']

# Drop rows with the specified tickers
df_filtered = df[~df['ticker'].isin(tickers_to_drop)]

# Save the filtered data to a new CSV file
df_filtered.to_csv('data_files/option_data_no_BACC.csv', index=False)