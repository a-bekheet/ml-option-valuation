import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_numeric_columns(df: pd.DataFrame):
    """
    Return a list of numeric columns to scale, excluding
    those you wish to ignore (like object or date/time).
    """
    # First, pick all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Decide which columns to EXCLUDE from scaling:
    # For example, if you don't want to scale day_of_week_sin/cos since they're already in [-1,1],
    # you could remove them from the list below.
    exclude = [
        'day_of_week_sin', 'day_of_week_cos',
        'day_of_month_sin', 'day_of_month_cos',
        'day_of_year_sin', 'day_of_year_cos',
        'inTheMoney'
    ]
    
    # Alternatively, if you DO want to scale them, just leave them alone.
    
    # Also exclude booleans if you haven't converted them to numeric 0/1 yet:
    # e.g., inTheMoney -> can be included if you've already cast True/False -> 1/0.
    
    # Filter out the excluded columns
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    return numeric_cols

def compute_scaler(file_path, chunksize=200_000, numeric_cols=None):
    """
    1st pass: read CSV in chunks, partial_fit to compute global means & std devs.
    """
    scaler = StandardScaler()
    
    first_chunk = True
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        if first_chunk and numeric_cols is None:
            # Identify numeric columns from the first chunk
            numeric_cols = get_numeric_columns(chunk)
            first_chunk = False
        
        # Convert bool columns if needed, e.g. chunk['inTheMoney'] = chunk['inTheMoney'].astype(int)
        if 'inTheMoney' in chunk.columns and chunk['inTheMoney'].dtype == bool:
            chunk['inTheMoney'] = chunk['inTheMoney'].astype(int)
        
        # Drop rows with NaN in numeric columns or handle them (impute, etc.)
        chunk = chunk.dropna(subset=numeric_cols)
        
        scaler.partial_fit(chunk[numeric_cols])
        if (i+1) % 10 == 0:
            print(f"Partial fit on {chunksize*(i+1)} rows ...")
    
    return scaler, numeric_cols

def transform_and_save(file_path, output_path, scaler, numeric_cols, chunksize=200_000):
    """
    2nd pass: transform numeric columns using the fitted scaler, write out CSV in chunks.
    """
    first_chunk = True
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        # Convert bool columns if needed
        if 'inTheMoney' in chunk.columns and chunk['inTheMoney'].dtype == bool:
            chunk['inTheMoney'] = chunk['inTheMoney'].astype(int)
        
        # Drop rows with NaN in numeric columns or handle them
        chunk = chunk.dropna(subset=numeric_cols)
        
        # Scale numeric columns
        chunk[numeric_cols] = scaler.transform(chunk[numeric_cols])
        
        # Write out
        mode = 'w' if first_chunk else 'a'
        header = True if first_chunk else False
        chunk.to_csv(output_path, mode=mode, header=header, index=False)
        first_chunk = False
        
        if (i+1) % 10 == 0:
            print(f"Transformed & saved {chunksize*(i+1)} rows...")

def main():
    file_path = "/Users/bekheet/dev/option-ml-prediction/data_files/option_data_time_encoded.csv"
    output_path = "/Users/bekheet/dev/option-ml-prediction/data_files/option_data_scaled.csv"
    
    print("Computing scaler...")
    scaler, numeric_cols = compute_scaler(file_path, chunksize=200_000)
    
    print("Transforming and saving data...")
    transform_and_save(file_path, output_path, scaler, numeric_cols, chunksize=200_000)
    
    print("All done! Normalized dataset is at:", output_path)

if __name__ == "__main__":
    main()
