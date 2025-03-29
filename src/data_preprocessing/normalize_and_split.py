"""
Module for processing financial data with per-ticker normalization and original data recovery
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Set

def get_numeric_columns(df: pd.DataFrame, exclude: Optional[Set[str]] = None) -> List[str]:
    """Return numeric columns to scale, excluding specified columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude:
        numeric_cols = [col for col in numeric_cols if col not in exclude]
    logging.debug(f"Identified numeric columns for scaling: {numeric_cols}")
    return numeric_cols

def compute_scaler_for_ticker(
    df: pd.DataFrame, 
    numeric_cols: List[str]
) -> Tuple[StandardScaler, Dict[str, Dict[str, float]]]:
    """
    Compute scaler and scaling parameters for a single ticker's data
    
    Args:
        df: DataFrame containing single ticker data
        numeric_cols: Numeric columns to scale
        
    Returns:
        Tuple of (fitted scaler, scaling parameters dictionary)
    """
    scaler = StandardScaler()
    
    # Remove NaN values from numeric columns
    df_clean = df.dropna(subset=numeric_cols)
    
    # Fit the scaler
    scaler.fit(df_clean[numeric_cols])
    
    # Store scaling parameters for later recovery
    scaling_params = {}
    for i, col in enumerate(numeric_cols):
        scaling_params[col] = {
            "mean": scaler.mean_[i],
            "scale": scaler.scale_[i]
        }
    
    return scaler, scaling_params

def transform_ticker_data(
    df: pd.DataFrame, 
    scaler: StandardScaler, 
    numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Apply scaling transformation to a ticker's data
    
    Args:
        df: DataFrame to transform
        scaler: Fitted StandardScaler
        numeric_cols: Numeric columns to scale
        
    Returns:
        Transformed DataFrame
    """
    df_result = df.copy()
    
    # Only transform rows without NaN values in numeric columns
    mask = ~df[numeric_cols].isna().any(axis=1)
    df_result.loc[mask, numeric_cols] = scaler.transform(df[numeric_cols].loc[mask])
    
    return df_result

# Add new chunked processing functions while keeping original names

def split_and_normalize_by_ticker(
    input_file: str,
    output_dir: str,
    ticker_column: str = 'ticker',
    exclude_columns: Optional[Set[str]] = None,
    chunksize: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Split data by ticker, normalize each ticker independently, and save scaling parameters
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save output files
        ticker_column: Column containing ticker symbols
        exclude_columns: Set of columns to exclude from scaling
        chunksize: Number of rows to process per chunk (if None, load entire file)
        
    Returns:
        Dictionary with ticker information
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a directory for scaling parameters
    scaling_dir = os.path.join(output_dir, 'scaling_params')
    Path(scaling_dir).mkdir(parents=True, exist_ok=True)
    
    # Default exclude set for cyclical and binary features
    if exclude_columns is None:
        exclude_columns = {
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'inTheMoney'
        }
    
    result = {}
    
    # Handle processing based on whether chunking is requested
    if chunksize is None:
        # Original in-memory processing
        # Read the data
        df = pd.read_csv(input_file)
        
        # Get unique tickers
        tickers = df[ticker_column].unique()
        
        # Process each ticker
        for ticker in tickers:
            # Extract ticker data
            ticker_data = df[df[ticker_column] == ticker].copy()
            ticker_count = len(ticker_data)
            
            # Skip if not enough data
            if ticker_count < 10:
                continue
            
            # Get numeric columns
            numeric_cols = get_numeric_columns(ticker_data, exclude_columns)
            
            # Compute scaler and parameters
            scaler, scaling_params = compute_scaler_for_ticker(ticker_data, numeric_cols)
            
            # Transform the data
            normalized_data = transform_ticker_data(ticker_data, scaler, numeric_cols)
            
            # Save normalized data
            normalized_file = os.path.join(output_dir, f"{ticker}_normalized.csv")
            normalized_data.to_csv(normalized_file, index=False)
            
            # Save scaling parameters for recovery
            params_file = os.path.join(scaling_dir, f"{ticker}_scaling_params.json")
            with open(params_file, 'w') as f:
                json.dump(scaling_params, f, indent=2)
            
            # Store information
            result[ticker] = {
                'file_path': normalized_file,
                'count': ticker_count,
                'scaling_params': params_file
            }
    else:
        # Chunked processing for large files
        # First pass: get unique tickers
        tickers = set()
        for chunk in pd.read_csv(input_file, chunksize=chunksize, usecols=[ticker_column]):
            tickers.update(chunk[ticker_column].unique())
        
        tickers = sorted(tickers)
        
        # Second pass: process each ticker
        for ticker in tickers:
            # Initialize data collection for this ticker
            all_ticker_data = []
            
            # Collect all data for this ticker
            for chunk in pd.read_csv(input_file, chunksize=chunksize):
                ticker_chunk = chunk[chunk[ticker_column] == ticker]
                if not ticker_chunk.empty:
                    all_ticker_data.append(ticker_chunk)
            
            if not all_ticker_data:
                continue
                
            # Combine all chunks for this ticker
            ticker_data = pd.concat(all_ticker_data)
            ticker_count = len(ticker_data)
            
            # Skip if not enough data
            if ticker_count < 10:
                continue
            
            # Get numeric columns
            numeric_cols = get_numeric_columns(ticker_data, exclude_columns)
            
            # Compute scaler and parameters
            scaler, scaling_params = compute_scaler_for_ticker(ticker_data, numeric_cols)
            
            # Transform the data
            normalized_data = transform_ticker_data(ticker_data, scaler, numeric_cols)
            
            # Save normalized data
            normalized_file = os.path.join(output_dir, f"{ticker}_normalized.csv")
            normalized_data.to_csv(normalized_file, index=False)
            
            # Save scaling parameters for recovery
            params_file = os.path.join(scaling_dir, f"{ticker}_scaling_params.json")
            with open(params_file, 'w') as f:
                json.dump(scaling_params, f, indent=2)
            
            # Store information
            result[ticker] = {
                'file_path': normalized_file,
                'count': ticker_count,
                'scaling_params': params_file
            }
    
    # Save ticker metadata
    metadata = pd.DataFrame([
        {
            'ticker': ticker,
            'file_path': info['file_path'],
            'count': info['count'],
            'scaling_params': info['scaling_params']
        }
        for ticker, info in result.items()
    ])
    
    metadata_file = os.path.join(output_dir, 'ticker_metadata.csv')
    metadata.to_csv(metadata_file, index=False)
    
    return result

def recover_original_data(
    normalized_file: str,
    scaling_params_file: str,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Recover original data from normalized data using scaling parameters
    
    Args:
        normalized_file: Path to normalized CSV file
        scaling_params_file: Path to scaling parameters JSON file
        output_file: Path to save recovered data (optional)
        
    Returns:
        DataFrame with recovered original data
    """
    # Read normalized data
    df = pd.read_csv(normalized_file)
    
    # Read scaling parameters
    with open(scaling_params_file, 'r') as f:
        scaling_params = json.load(f)
    
    # Recover original data
    df_recovered = df.copy()
    
    # Apply inverse transformation to each column
    for col, params in scaling_params.items():
        # Skip if column not in DataFrame
        if col not in df.columns:
            continue
            
        # Get parameters
        mean = params['mean']
        scale = params['scale']
        
        # Inverse transform: X_orig = X_scaled * scale + mean
        mask = ~df[col].isna()
        df_recovered.loc[mask, col] = df.loc[mask, col] * scale + mean
    
    # Save if output file provided
    if output_file:
        df_recovered.to_csv(output_file, index=False)
    
    return df_recovered

