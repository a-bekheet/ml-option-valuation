"""
This module provides functionality for analyzing and optimizing the memory usage of a pandas DataFrame. 
It includes functions to perform detailed memory analysis of DataFrame columns, suggest optimization 
strategies based on column statistics, and handle specific data types.
Functions:
    - analyze_memory_usage(df): Analyzes memory usage of DataFrame columns, providing detailed statistics 
      such as data type, memory usage in MB, percentage of total memory, unique value count, and min/max 
      values for numeric columns. It also prints a summary and per-column analysis with optimization suggestions.
    - suggest_optimization(column, stats, series): Suggests optimization strategies for a given column based 
      on its statistics. It provides recommendations for converting data types to more memory-efficient types 
      where applicable.
The module also includes a script to load a CSV file, analyze its memory usage, and print the analysis results. 
If the file is not found or an error occurs during loading, appropriate error messages are displayed.

"""

import pandas as pd
import numpy as np
import os

def analyze_memory_usage(df):
    """
    Detailed memory analysis of DataFrame columns with type-specific handling
    """
    # Get memory usage of each column
    memory_usage = df.memory_usage(deep=True)
    
    # Initialize analysis dictionary
    analysis = {
        'dtype': df.dtypes,
        'memory_mb': memory_usage / 1e6,  # Convert to MB
        'memory_pct': memory_usage / memory_usage.sum() * 100,
        'unique_count': df.nunique()
    }
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame(analysis)
    
    # Add min/max for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis_df.loc[numeric_cols, 'min_value'] = df[numeric_cols].min()
    analysis_df.loc[numeric_cols, 'max_value'] = df[numeric_cols].max()
    
    # Sort by memory usage
    analysis_df = analysis_df.sort_values('memory_mb', ascending=False)
    
    print("\nMemory Usage Analysis:")
    print("=" * 80)
    print(f"Total Memory Usage: {analysis_df['memory_mb'].sum():.2f} MB")
    print("\nPer Column Analysis:")
    print("-" * 80)
    
    for idx, row in analysis_df.iterrows():
        print(f"\nColumn: {idx}")
        print(f"Type: {row['dtype']}")
        print(f"Memory: {row['memory_mb']:.2f} MB ({row['memory_pct']:.1f}%)")
        print(f"Unique Values: {row['unique_count']}")
        
        # Print range only for numeric columns
        if idx in numeric_cols:
            print(f"Range: {row['min_value']} to {row['max_value']}")
        
        # Sample values for non-numeric columns
        else:
            sample_values = df[idx].dropna().head(3).tolist()
            print(f"Sample Values: {sample_values}")
        
        # Suggest optimizations
        suggest_optimization(idx, row, df[idx])

def suggest_optimization(column, stats, series):
    """Suggest optimization strategies based on column statistics"""
    dtype = str(stats['dtype'])
    
    if 'float' in dtype:
        if pd.notnull(stats.get('max_value')):
            max_val = stats['max_value']
            min_val = stats['min_value']
            if max_val < 65500 and min_val > -65500:
                if series.round(4).eq(series).all():
                    print("➤ Could be converted to integer type")
                else:
                    print("➤ Could be converted to float32")
            else:
                print("➤ Needs to remain float64 due to range")
            
    elif 'int' in dtype:
        if pd.notnull(stats.get('max_value')):
            max_val = stats['max_value']
            min_val = stats['min_value']
            if max_val < 255 and min_val >= 0:
                print("➤ Could be converted to uint8")
            elif max_val < 65535 and min_val >= 0:
                print("➤ Could be converted to uint16")
            elif max_val < 4294967295 and min_val >= 0:
                print("➤ Could be converted to uint32")
            
    elif dtype == 'object':
        if stats['unique_count'] == 1:
            print("➤ Single value - could be converted to category")
        elif stats['unique_count'] < 50:
            print("➤ Low cardinality - could be converted to category")
        elif 'date' in column.lower():
            print("➤ Could be converted to datetime64")

# Load and analyze data
print("Current working directory:", os.getcwd())
try:
    df = pd.read_csv('data_files/option_data_with_headers_cleaned.csv')  # Adjust path as needed
    analyze_memory_usage(df)
except FileNotFoundError as e:
    print(f"Error: Could not find the file. Please check the path. Error details: {e}")
except Exception as e:
    print(f"An error occurred: {e}")