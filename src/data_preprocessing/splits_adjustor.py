import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

def adjust_for_stock_splits(
    data_path: str, 
    output_path: Optional[str] = None,
    return_dataframe: bool = True
) -> Optional[pd.DataFrame]:
    """
    Adjust stock and option data for historical stock splits.
    
    Args:
        data_path: Path to input CSV file
        output_path: Path to output CSV file (default: input filename with '_split_adjusted' suffix)
        return_dataframe: Whether to return the adjusted DataFrame
        
    Returns:
        Adjusted DataFrame if return_dataframe is True, otherwise None
    """
    # Set output path if not provided
    if output_path is None:
        output_path = data_path.replace('.csv', '_split_adjusted.csv')
    
    # Manually verified splits with exact ratios
    verified_splits = {
        'AMZN': {
            'date': '2022-06-06',
            'ratio': 20.0,  # 20:1 split
            'type': 'forward'
        },
        'GOOGL': {
            'date': '2022-07-18',
            'ratio': 20.0,  # 20:1 split
            'type': 'forward'
        },
        'TSLA': {
            'date': '2022-08-25',
            'ratio': 3.0,   # 3:1 split
            'type': 'forward'
        },
        'SHOP': {
            'date': '2022-07-04',
            'ratio': 10.0,  # 10:1 split
            'type': 'forward'
        },
        'CGC': {
            'date': '2023-12-20',
            'ratio': 10.0,  # 1:10 reverse split (stored as multiplier)
            'type': 'reverse'
        }
    }
    
    # Features to adjust
    price_features = [
        'strike', 'lastPrice', 'bid', 'ask', 'change',
        'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
        'strikeDelta', 'stockClose_ewm_5d', 'stockClose_ewm_15d',
        'stockClose_ewm_45d', 'stockClose_ewm_135d'
    ]
    
    volume_features = [
        'volume', 'openInterest', 'stockVolume'
    ]
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert date columns
    date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].str.split('+').str[0])
    
    # Create copy for adjustment
    df_adjusted = df.copy()
    
    # Convert numeric columns to float64
    for col in price_features + volume_features:
        if col in df_adjusted.columns:
            df_adjusted[col] = df_adjusted[col].astype('float64')
    
    # Helper function to verify adjustments
    def verify_adjustment_metrics(df: pd.DataFrame, ticker: str, split_date: pd.Timestamp, 
                                 feature: str, window_days: int = 3) -> Tuple[float, float, float]:
        """Calculate continuity metrics for a specific feature around a split date."""
        ticker_data = df[df['ticker'] == ticker]
        
        pre_split = ticker_data[
            (ticker_data['quoteDate'] < split_date) & 
            (ticker_data['quoteDate'] >= split_date - timedelta(days=window_days))
        ]
        
        post_split = ticker_data[
            (ticker_data['quoteDate'] > split_date) & 
            (ticker_data['quoteDate'] <= split_date + timedelta(days=window_days))
        ]
        
        if len(pre_split) == 0 or len(post_split) == 0:
            return 0, 0, float('nan')
        
        pre_value = pre_split[feature].median()
        post_value = post_split[feature].median()
        
        if pre_value > 0 and post_value > 0:
            discontinuity = abs(1 - (pre_value / post_value))
        else:
            discontinuity = float('nan')
            
        return pre_value, post_value, discontinuity
    
    # Process each split
    for ticker, split_info in verified_splits.items():
        split_date = pd.to_datetime(split_info['date'])
        split_ratio = split_info['ratio']
        split_type = split_info['type']
        
        # Get pre-split data mask
        ticker_mask = df_adjusted['ticker'] == ticker
        pre_split_mask = ticker_mask & (df_adjusted['quoteDate'] < split_date)
        affected_records = pre_split_mask.sum()
        
        if affected_records == 0:
            continue
        
        # Apply adjustments based on split type
        adjustment_ratio = split_ratio if split_type == 'forward' else (1 / split_ratio)
        
        # Adjust price features
        for feature in price_features:
            if feature in df_adjusted.columns:
                mask = pre_split_mask & df_adjusted[feature].notna()
                df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] / adjustment_ratio
        
        # Adjust volume features (inverse adjustment)
        for feature in volume_features:
            if feature in df_adjusted.columns:
                mask = pre_split_mask & df_adjusted[feature].notna()
                df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] * adjustment_ratio
    
    # Save adjusted data
    df_adjusted.to_csv(output_path, index=False)
    
    # Return dataframe if requested
    if return_dataframe:
        return df_adjusted
    return None