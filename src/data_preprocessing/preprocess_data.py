import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import os
import json
from drops import Drop
from dtypes_adjustor import datatype_adjustor
from splits_adjustor import adjust_for_stock_splits
from cyclic_encoding import encode_dataframe
from normalize_and_split import get_numeric_columns, compute_scaler_for_ticker, transform_ticker_data
from normalize_and_split_utils import scale_dataset, split_data_by_ticker

def preprocess_data(
    input_file: str, 
    output_file: str,
    splits_file: str, 
    drop_config: Dict[str, List[str]],
    normalize_per_ticker: bool = True
) -> None:
    # Adjust data types
    datatype_adjustor(input_file, output_file)
    
    # Adjust for stock splits
    adjust_for_stock_splits(output_file, splits_file)
    
    # Drop columns
    drop_config = Drop(drop_config)
    drop_config.drop_columns(output_file, output_file)
    
    # Encode cyclical features
    encode_dataframe(output_file, output_file)
    
    if normalize_per_ticker:
        # Create directory for split normalized files
        split_dir = output_file.replace('.csv', '_by_ticker')
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        
        # Create directory for scaling parameters
        params_dir = os.path.join(split_dir, 'scaling_params')
        Path(params_dir).mkdir(parents=True, exist_ok=True)
        
        # Read the processed data
        df = pd.read_csv(output_file)
        
        # Define columns to exclude from scaling
        exclude_cols = {
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'inTheMoney'
        }
        
        # Process each ticker separately
        tickers = df['ticker'].unique()
        ticker_info = {}
        
        for ticker in tickers:
            # Extract ticker data
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # Skip if not enough data
            if len(ticker_data) < 10:
                continue
            
            # Get numeric columns for this ticker
            numeric_cols = get_numeric_columns(ticker_data, exclude_cols)
            
            # Compute scaler and parameters
            scaler, scaling_params = compute_scaler_for_ticker(ticker_data, numeric_cols)
            
            # Transform the data
            normalized_data = transform_ticker_data(ticker_data, scaler, numeric_cols)
            
            # Save normalized data
            normalized_file = os.path.join(split_dir, f"{ticker}_normalized.csv")
            normalized_data.to_csv(normalized_file, index=False)
            
            # Save scaling parameters
            params_file = os.path.join(params_dir, f"{ticker}_scaling_params.json")
            with open(params_file, 'w') as f:
                json.dump(scaling_params, f, indent=2)
            
            # Store info
            ticker_info[ticker] = {
                'file_path': normalized_file,
                'count': len(ticker_data),
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
            for ticker, info in ticker_info.items()
        ])
        
        metadata_file = os.path.join(split_dir, 'ticker_metadata.csv')
        metadata.to_csv(metadata_file, index=False)
    else:
        # Global normalization (original approach)
        scale_dataset(output_file, output_file)
        
        # Split data by ticker
        split_data_by_ticker(output_file, output_file.replace('.csv', '_split_new'), 'ticker')
    
    """
    TODO:
    - Compute rolling windows for stock price features
    - Add options greeks
    """