"""
Module for encoding temporal features in pandas DataFrames
"""
import pandas as pd
import numpy as np
from typing import Optional, Union, List

def encode_cyclical_feature(df: pd.DataFrame, column: str, period: int, 
                            offset: int = 1, keep_original: bool = True) -> pd.DataFrame:
    """Encode a single cyclical feature with sine and cosine transforms"""
    sin_col = f"{column}_sin"
    cos_col = f"{column}_cos"
    
    # Apply sine and cosine transformations (offset typically used to convert 1-based to 0-based)
    df[sin_col] = np.sin(2 * np.pi * (df[column] - offset) / period)
    df[cos_col] = np.cos(2 * np.pi * (df[column] - offset) / period)
    
    # Remove original column if not needed
    if not keep_original and column not in df.columns:
        df.drop(columns=[column], inplace=True)
        
    return df

def encode_temporal_features(df: pd.DataFrame, 
                            date_column: str = 'lastTradeDate',
                            keep_original_dates: bool = True,
                            keep_original_features: bool = True) -> pd.DataFrame:
    """
    Encode date/time features using cyclical encoding.
    
    Args:
        df: DataFrame with date column
        date_column: Name of the date column to encode
        keep_original_dates: Whether to keep the original date column
        keep_original_features: Whether to keep the extracted date features
        
    Returns:
        DataFrame with encoded temporal features
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
        result[date_column] = pd.to_datetime(result[date_column])
    
    # Extract temporal features
    result['day_of_week'] = result[date_column].dt.dayofweek    # Monday=0, Sunday=6
    result['day_of_month'] = result[date_column].dt.day         # 1-31
    result['day_of_year'] = result[date_column].dt.dayofyear    # 1-365/366
    
    # Apply cyclical encoding to each feature
    result = encode_cyclical_feature(result, 'day_of_week', 7, offset=0, keep_original=keep_original_features)
    result = encode_cyclical_feature(result, 'day_of_month', 31, offset=1, keep_original=keep_original_features)
    result = encode_cyclical_feature(result, 'day_of_year', 365, offset=1, keep_original=keep_original_features)
    
    # Remove original date column if not needed
    if not keep_original_dates:
        result.drop(columns=[date_column], inplace=True)
        
    return result

def encode_dataframe(input_path: Optional[str] = None,
                    output_path: Optional[str] = None,
                    df: Optional[pd.DataFrame] = None,
                    date_column: str = 'lastTradeDate',
                    keep_original_dates: bool = True,
                    keep_original_features: bool = True) -> pd.DataFrame:
    """
    Process a DataFrame from file or directly and encode temporal features
    
    Args:
        input_path: Path to CSV file (alternative to df)
        output_path: Path to save encoded DataFrame (optional)
        df: DataFrame to process (alternative to input_path)
        date_column: Name of the date column to encode
        keep_original_dates: Whether to keep the original date column
        keep_original_features: Whether to keep the extracted date features
        
    Returns:
        DataFrame with encoded temporal features
    """
    # Load data if path provided
    if df is None and input_path:
        df = pd.read_csv(input_path)
    elif df is None:
        raise ValueError("Either df or input_path must be provided")
        
    # Encode temporal features
    encoded_df = encode_temporal_features(
        df, 
        date_column=date_column,
        keep_original_dates=keep_original_dates,
        keep_original_features=keep_original_features
    )
    
    # Save to file if path provided
    if output_path:
        encoded_df.to_csv(output_path, index=False)
        
    return encoded_df