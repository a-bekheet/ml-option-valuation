import pandas as pd

# Load the CSV file
df = pd.read_csv('data_files/option_data_compressed.csv')

def extract_ticker(symbol):
    # Extract ticker from contractSymbol (e.g., 'AAPL220218C00150000' -> 'AAPL')
    return ''.join(filter(str.isalpha, symbol.split('220')[0]))

# Add ticker column
df['ticker'] = df['contractSymbol'].apply(extract_ticker)

def count_ticker_rows(df):
    """
    Count the number of rows for each ticker and print them in a table
    """
    ticker_counts = df['ticker'].value_counts().reset_index()
    ticker_counts.columns = ['Ticker', 'Number of Rows']
    print("Ticker Counts:")
    print(ticker_counts.to_string(index=False))

# Call the function to print the table of ticker counts
count_ticker_rows(df)

# Drop rows where 'stockClose_ewm_5d' is missing
df_cleaned = df.dropna(subset=['stockClose_ewm_5d'])

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('data_files/option_data_no_BAC.csv', index=False)