import pandas as pd
import re

def extract_ticker(symbol):
    # Extract ticker from contractSymbol (e.g., 'AAPL220218C00150000' -> 'AAPL')
    ticker = re.split(r'\d', symbol)[0]
    return ticker


def prelim_dropping(input_file, output_file, ticker_list = ['BAC', 'C']):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Add ticker column
    df['ticker'] = df['contractSymbol'].apply(extract_ticker)

    # Drop rows with the specified tickers
    df_filtered = df[~df['ticker'].isin(ticker_list)]

    # Drop rows where 'stockClose_ewm_5d' is missing
    df_filtered = df.dropna(subset=['stockClose_ewm_5d'])

    # Save the filtered data to a new CSV file
    df_filtered.to_csv(output_file, index=False)