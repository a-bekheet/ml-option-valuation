import pandas as pd
import numpy as np
from datetime import datetime

def analyze_csv_columns(filepath='data_files/option_data_compressed.csv'):
    """Analyze column properties of the option data CSV"""
    # Define date columns and dtypes
    date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
    
    # Read the CSV with proper type parsing
    df = pd.read_csv(
        filepath,
        parse_dates=date_columns,
        dtype={
            'contractSymbol': 'category',
            'contractSize': 'category',
            'currency': 'category',
            'inTheMoney': 'bool',
            'daysToExpiry': 'uint16',
            'stockVolume': 'uint32',
            'volume': 'uint32',
            'openInterest': 'uint32'
        }
    )
    
    analysis = []
    
    for col in df.columns:
        # Calculate basic statistics
        missing = df[col].isna().sum()
        total = len(df)
        present = total - missing
        pct_present = (present / total) * 100
        
        # Calculate unique values
        unique_count = df[col].nunique()
        pct_unique = (unique_count / total) * 100
        
        # Get sample values (non-null)
        sample_values = df[col].dropna().head(3).tolist()
        
        # Determine data type and statistics
        dtype = str(df[col].dtype)
        
        # Handle different types of columns
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            stats = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        elif pd.api.types.is_datetime64_dtype(df[col]):
            stats = {
                'min_date': df[col].min(),
                'max_date': df[col].max(),
                'date_range_days': (df[col].max() - df[col].min()).days
            }
        elif pd.api.types.is_categorical_dtype(df[col]):
            stats = {
                'categories_count': len(df[col].cat.categories),
                'memory_usage_mb': df[col].memory_usage(deep=True) / 1024**2
            }
        else:
            stats = None
            
        # Add recommendation based on data quality
        if pct_present > 99.5:
            recommendation = 'Keep'
        elif pct_present > 98:
            recommendation = 'Review'
        else:
            recommendation = 'Consider Dropping'
            
        analysis.append({
            'Column Name': col,
            'Data Type': dtype,
            'Present Count': present,
            'Missing Count': missing,
            'Percentage Present': round(pct_present, 2),
            'Unique Values': unique_count,
            'Percentage Unique': round(pct_unique, 2),
            'Sample Values': str(sample_values)[:50] + '...' if len(str(sample_values)) > 50 else str(sample_values),
            'Statistics': stats,
            'Recommendation': recommendation
        })
    
    # Convert to DataFrame
    analysis_df = pd.DataFrame(analysis)
    
    # Print summary
    print("\nColumn Analysis Summary:")
    print("-" * 80)
    summary = analysis_df[['Column Name', 'Data Type', 'Percentage Present', 
                          'Percentage Unique', 'Recommendation']]
    print(summary.to_string(index=False))
    
    # Print data quality concerns
    print("\nData Quality Concerns:")
    print("-" * 80)
    concerns = analysis_df[analysis_df['Percentage Present'] < 99.5]
    if len(concerns) > 0:
        for _, row in concerns.iterrows():
            print(f"Column '{row['Column Name']}' has {row['Missing Count']} missing values "
                  f"({100 - row['Percentage Present']:.2f}% missing)")
    else:
        print("No major data quality concerns found")
    
    # Print high cardinality columns
    print("\nHigh Cardinality Columns (>50% unique values):")
    print("-" * 80)
    high_cardinality = analysis_df[analysis_df['Percentage Unique'] > 50]
    for _, row in high_cardinality.iterrows():
        print(f"Column '{row['Column Name']}' has {row['Unique Values']} unique values "
              f"({row['Percentage Unique']:.2f}% unique)")
    
    # Add memory usage analysis
    print("\nMemory Usage Analysis:")
    print("-" * 80)
    total_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Total Memory Usage: {total_memory:.2f} MB")
    
    # Save detailed analysis
    analysis_df.to_csv('option_data_compressed_detailed_analysis.csv', index=False)
    
    return analysis_df

# Run the analysis
analysis = analyze_csv_columns()