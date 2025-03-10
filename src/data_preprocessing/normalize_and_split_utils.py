import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Set

def split_data_by_ticker(
    input_file: str, 
    output_dir: str,
    ticker_column: str = 'ticker'
) -> Dict[str, Dict[str, Any]]:
    """
    Split a CSV file containing multiple tickers into separate files by ticker.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the split files
        ticker_column: Name of the column containing ticker symbols
        
    Returns:
        Dictionary mapping tickers to their file paths and data counts
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the data
    df = pd.read_csv(input_file)
    
    # Get unique tickers and their counts
    ticker_counts = df[ticker_column].value_counts().sort_index()
    
    ticker_files = {}
    
    # Process each ticker
    for ticker in ticker_counts.index:
        ticker_data = df[df[ticker_column] == ticker].copy()
        output_file = os.path.join(output_dir, f"option_data_scaled_{ticker}.csv")
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
    
    return ticker_files

def get_numeric_columns(df: pd.DataFrame, exclude: Optional[Set[str]] = None) -> List[str]:
    """Return numeric columns to scale, excluding specified columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude:
        numeric_cols = [col for col in numeric_cols if col not in exclude]
    return numeric_cols

def compute_scaler(
    file_path: str, 
    chunksize: int = 200_000, 
    numeric_cols: Optional[List[str]] = None,
    exclude: Optional[Set[str]] = None
) -> Tuple[StandardScaler, List[str]]:
    """Compute global scaler from CSV in chunks"""
    scaler = StandardScaler()
    first_chunk = True
    
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        if first_chunk and numeric_cols is None:
            numeric_cols = get_numeric_columns(chunk, exclude)
            first_chunk = False
            
        chunk = chunk.dropna(subset=numeric_cols)
        scaler.partial_fit(chunk[numeric_cols])
        
    return scaler, numeric_cols

def transform_and_save(
    file_path: str, 
    output_path: str, 
    scaler: StandardScaler, 
    numeric_cols: List[str], 
    chunksize: int = 200_000
) -> None:
    """Transform data using fitted scaler and save to CSV"""
    first_chunk = True
    
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunk = chunk.dropna(subset=numeric_cols)
        chunk[numeric_cols] = scaler.transform(chunk[numeric_cols])
        
        chunk.to_csv(
            output_path, 
            mode='w' if first_chunk else 'a', 
            header=first_chunk, 
            index=False
        )
        first_chunk = False

def scale_dataset(
    input_path: str, 
    output_path: str, 
    chunksize: int = 200_000,
    exclude_columns: Optional[Set[str]] = None
) -> StandardScaler:
    """Scale a dataset and return the fitted scaler"""
    scaler, numeric_cols = compute_scaler(input_path, chunksize, exclude=exclude_columns)
    transform_and_save(input_path, output_path, scaler, numeric_cols, chunksize)
    return scaler