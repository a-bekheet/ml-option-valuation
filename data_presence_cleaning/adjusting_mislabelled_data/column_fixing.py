import numpy as np
import pandas as pd

def rearrange_option_data():
    # Load the raw numpy array
    data = np.load('concatenated_data.npy', allow_pickle=True)
    
    # Column names in their correct order
    column_names = [
        'contractSymbol',      # 0
        'lastTradeDate',       # 1
        'strike',              # 2
        'lastPrice',           # 3
        'bid',                 # 4
        'ask',                 # 5
        'change',              # 6
        'percentChange',       # 7
        'volume',              # 8
        'openInterest',        # 9
        'impliedVolatility',   # 10
        'inTheMoney',          # 11
        'contractSize',        # 12
        'currency',            # 13
        'quoteDate',           # 14
        'expiryDate',          # 15
        'daysToExpiry',        # 16
        'stockVolume',         # 17
        'stockClose',          # 18
        'stockAdjClose',       # 19
        'stockOpen',           # 20
        'stockHigh',           # 21
        'stockLow',            # 22
        'strikeDelta',         # 23
        'stockClose_ewm_5d',   # 24
        'stockClose_ewm_15d',  # 25
        'stockClose_ewm_45d',  # 26
        'stockClose_ewm_135d'  # 27
    ]
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Set column names
    df.columns = column_names
    
    # Save the rearranged data
    np.save('option_data_with_headers.npy', df.to_numpy())
    
    # Also save as CSV for easy inspection
    df.to_csv('option_data_with_headers.csv', index=False)
    
    return df

# Execute the rearrangement
df = rearrange_option_data()

# Print some information about the result
print("\nDataset Info:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)