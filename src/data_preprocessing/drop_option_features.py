"""
This section drops the four stockClose_ewm columns as they are only absent for one ticker.
"""
import pandas as pd

# Load the CSV file
df = pd.read_csv('data_files/option_data_with_headers.csv')

# Drop rows where 'stockClose_ewm_5d' is missing
df_cleaned = df.dropna(subset=['stockClose_ewm_5d'])

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('data_files/option_data_with_headers_cleaned.csv', index=False)
