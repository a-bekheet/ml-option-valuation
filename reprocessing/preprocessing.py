#!/usr/bin/env python3
"""
Data Pipeline for Option Data:
- Rearrange raw NumPy data to a DataFrame with proper headers.
- Clean data (drop rows missing critical values).
- Analyze column completeness and memory usage.
- Optimize datatypes (with sample validation) and process full dataset in chunks.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import psutil

warnings.filterwarnings('ignore')

# ==================== Constants / File Paths ====================
RAW_DATA_FILE       = 'reprocessing/concatenated_data.npy'
HEADERED_CSV_FILE   = 'reprocessing/option_data_with_headers_new.csv'
CLEANED_CSV_FILE    = os.path.join('data_files', 'option_data_with_headers_new_cleaned.csv')
COMPRESSED_CSV_FILE = os.path.join('data_files', 'option_data_compressed.csv')

# ==================== Step 1: Rearranging Raw Data ====================
def rearrange_option_data(raw_file=RAW_DATA_FILE, output_csv=HEADERED_CSV_FILE):
    """
    Load raw numpy data, assign proper column headers, and save as CSV and npy.
    """
    print("Loading raw data and rearranging columns...")
    data = np.load(raw_file, allow_pickle=True)
    column_names = [
        'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
        'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility',
        'inTheMoney', 'contractSize', 'currency', 'quoteDate', 'expiryDate',
        'daysToExpiry', 'stockVolume', 'stockClose', 'stockAdjClose', 'stockOpen',
        'stockHigh', 'stockLow', 'strikeDelta', 'stockClose_ewm_5d',
        'stockClose_ewm_15d', 'stockClose_ewm_45d', 'stockClose_ewm_135d'
    ]
    df = pd.DataFrame(data, columns=column_names)
    # Save as npy and CSV for further processing/inspection
    np.save('option_data_with_headers.npy', df.to_numpy())
    df.to_csv(output_csv, index=False)
    print(f"Rearranged data saved to {output_csv}")
    return df

# ==================== Step 2: Clean the Data ====================
def clean_option_data(input_csv=HEADERED_CSV_FILE, output_csv=CLEANED_CSV_FILE):
    """
    Clean the option data by dropping rows missing a critical column.
    (Here, we drop rows missing 'stockClose_ewm_5d'.)
    """
    print("Cleaning data – dropping rows with missing 'stockClose_ewm_5d'...")
    df = pd.read_csv(input_csv)
    df_cleaned = df.dropna(subset=['stockClose_ewm_5d'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_cleaned.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv} with {len(df_cleaned)} rows.")
    return df_cleaned

# ==================== Step 3: Analyze Column Quality ====================
def analyze_columns(file_path=CLEANED_CSV_FILE):
    """
    Analyze column presence in the CSV and print quality and recommendation details.
    """
    print("Analyzing column data quality...")
    expected_columns = [
        'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
        'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility',
        'inTheMoney', 'contractSize', 'currency', 'quoteDate', 'expiryDate',
        'daysToExpiry', 'stockVolume', 'stockClose', 'stockAdjClose', 'stockOpen',
        'stockHigh', 'stockLow', 'strikeDelta', 'stockClose_ewm_5d',
        'stockClose_ewm_15d', 'stockClose_ewm_45d', 'stockClose_ewm_135d'
    ]
    df = pd.read_csv(file_path)
    presence_data = pd.DataFrame({
        'Column Name': expected_columns,
        'Present': [df[col].notna().sum() for col in expected_columns],
        'Missing': [df[col].isna().sum() for col in expected_columns]
    })
    presence_data['Percentage Present'] = (
        presence_data['Present'] / (presence_data['Present'] + presence_data['Missing']) * 100
    )
    presence_data['Data Type'] = [str(df[col].dtype) for col in expected_columns]
    presence_data['Recommendation'] = presence_data['Percentage Present'].apply(
        lambda x: 'Keep' if x > 99.5 else 'Review' if x > 98 else 'Consider Dropping'
    )
    print("\nColumn Analysis Summary:")
    print("-" * 80)
    print(presence_data.to_string(index=False))
    print("\nData Quality Concerns:")
    print("-" * 80)
    quality_issues = presence_data[presence_data['Percentage Present'] < 99.5]
    for _, row in quality_issues.iterrows():
        print(f"Column '{row['Column Name']}' has {row['Missing']} missing values "
              f"({100 - row['Percentage Present']:.2f}% missing)")
    presence_data.to_csv('column_analysis_recommendations.csv', index=False)
    return presence_data

# ==================== Step 4: Memory & Datatype Analysis ====================
def analyze_memory_usage(df):
    """
    Detailed memory analysis of DataFrame columns with type-specific handling.
    Skips the 'Index' metadata since it doesn't exist as a regular column.
    """
    memory_usage = df.memory_usage(deep=True)
    analysis = {
        'dtype': df.dtypes,
        'memory_mb': memory_usage / 1e6,  # Convert bytes to MB
        'memory_pct': memory_usage / memory_usage.sum() * 100,
        'unique_count': df.nunique()
    }
    analysis_df = pd.DataFrame(analysis)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis_df.loc[numeric_cols, 'min_value'] = df[numeric_cols].min()
    analysis_df.loc[numeric_cols, 'max_value'] = df[numeric_cols].max()
    analysis_df = analysis_df.sort_values('memory_mb', ascending=False)
    
    print("\nMemory Usage Analysis:")
    print("=" * 80)
    print(f"Total Memory Usage: {analysis_df['memory_mb'].sum():.2f} MB")
    print("\nPer Column Analysis:")
    print("-" * 80)
    
    for col, row in analysis_df.iterrows():
        print(f"\nColumn: {col}")
        print(f"  Type: {row['dtype']}")
        print(f"  Memory: {row['memory_mb']:.2f} MB ({row['memory_pct']:.1f}%)")
        print(f"  Unique Values: {row['unique_count']}")
        # Only show details if the key exists as a regular column
        if col in df.columns:
            if col in numeric_cols:
                print(f"  Range: {row['min_value']} to {row['max_value']}")
            else:
                sample_vals = df[col].dropna().head(3).tolist()
                print(f"  Sample Values: {sample_vals}")
            suggest_optimization(col, row, df[col])
        else:
            print("  (Index metadata, skipping sample values and optimization suggestions)")

def suggest_optimization(column, stats, series):
    """
    Based on the stats and data range, suggest datatype optimizations.
    """
    dtype = str(stats['dtype'])
    if 'float' in dtype:
        if pd.notnull(stats.get('max_value')):
            max_val = stats['max_value']
            min_val = stats['min_value']
            if max_val < 65500 and min_val > -65500:
                if series.round(4).equals(series):
                    print("  ➤ Could be converted to integer type")
                else:
                    print("  ➤ Could be converted to float32")
            else:
                print("  ➤ Needs to remain float64 due to range")
    elif 'int' in dtype:
        if pd.notnull(stats.get('max_value')):
            max_val = stats['max_value']
            min_val = stats['min_value']
            if max_val < 255 and min_val >= 0:
                print("  ➤ Could be converted to uint8")
            elif max_val < 65535 and min_val >= 0:
                print("  ➤ Could be converted to uint16")
            elif max_val < 4294967295 and min_val >= 0:
                print("  ➤ Could be converted to uint32")
    elif dtype == 'object':
        if stats['unique_count'] == 1:
            print("  ➤ Single value – consider converting to category")
        elif stats['unique_count'] < 50:
            print("  ➤ Low cardinality – consider converting to category")
        elif 'date' in column.lower():
            print("  ➤ Could be converted to datetime64")

# ==================== Step 5: Full Dataset Processing (Chunked) ====================
def process_full_dataset(input_file=CLEANED_CSV_FILE, output_file=COMPRESSED_CSV_FILE, chunksize=100000):
    """
    Process the full dataset in chunks: convert datatypes and parse dates,
    while tracking progress and memory usage.
    """
    process_obj = psutil.Process()
    initial_memory = process_obj.memory_info().rss / 1024**2
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    total_rows = sum(1 for _ in open(input_file)) - 1
    print(f"Total rows to process: {total_rows:,}")
    
    sample_df = pd.read_csv(input_file, nrows=5)
    print("\nInitial column dtypes:")
    for col in sample_df.columns:
        print(f"  {col}: {sample_df[col].dtype}")
    
    dtypes = {
        'strike': 'float32',
        'lastPrice': 'float32',
        'bid': 'float32',
        'ask': 'float32',
        'change': 'float32',
        'percentChange': 'float32',
        'impliedVolatility': 'float32',
        'contractSize': 'category',
        'currency': 'category',
        'stockClose': 'float32',
        'stockHigh': 'float32',
        'stockLow': 'float32',
        'stockOpen': 'float32',
        'stockVolume': 'float64',
        'volume': 'float64',
        'openInterest': 'float64'
    }
    date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
    
    def convert_types(chunk):
        int_conversions = {
            'stockVolume': 'uint32',
            'volume': 'uint32',
            'openInterest': 'uint32'
        }
        for col, dtype in int_conversions.items():
            chunk[col] = chunk[col].fillna(0).astype(dtype)
        for col in date_columns:
            chunk[col] = pd.to_datetime(chunk[col])
        # Convert daysToExpiry from a string like "4 days" to a uint16
        chunk['daysToExpiry'] = (chunk['daysToExpiry']
                                 .str.replace(' days', '')
                                 .fillna(0)
                                 .astype('uint16'))
        return chunk
    
    rows_processed = 0
    chunks_processed = 0
    start_time = time.time()
    first_chunk = True
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    for chunk in pd.read_csv(input_file, chunksize=chunksize, dtype=dtypes):
        chunk = convert_types(chunk)
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(output_file, mode=mode, header=header, index=False)
        if first_chunk:
            print("\nFirst chunk dtypes after conversion:")
            for col in chunk.columns:
                print(f"  {col}: {chunk[col].dtype}")
            first_chunk = False
        rows_processed += len(chunk)
        chunks_processed += 1
        if chunks_processed % 5 == 0:
            percent_complete = (rows_processed / total_rows) * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (rows_processed / total_rows)
            remaining_time = estimated_total_time - elapsed_time
            print(f"\nProgress Update:")
            print(f"  Processed: {rows_processed:,}/{total_rows:,} rows ({percent_complete:.1f}%)")
            print(f"  Memory usage: {process_obj.memory_info().rss / 1024**2:.1f} MB")
            print(f"  Time elapsed: {elapsed_time/60:.1f} minutes")
            print(f"  Estimated time remaining: {remaining_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print("\nProcessing Complete!")
    print("=" * 50)
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Final row count: {rows_processed:,}")
    print(f"  Final memory usage: {process_obj.memory_info().rss / 1024**2:.1f} MB")

# ==================== Step 6: Sample-Based Optimization Testing ====================
def create_sample_data(file_path, sample_size=1000, random_seed=42):
    """
    Create a sample dataset from the large CSV.
    """
    np.random.seed(random_seed)
    df_sample = pd.read_csv(file_path, nrows=sample_size)
    print(f"Created sample with {len(df_sample)} rows")
    return df_sample

def validate_transformed_data(original_df, optimized_df):
    """
    Validate data transformations between original and optimized DataFrames.
    """
    print("\nValidating Data Transformations:")
    print("=" * 80)
    validation_results = {}
    
    def check_date_columns():
        date_cols = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_cols:
            sample_orig = original_df[col].iloc[0]
            sample_opt = optimized_df[col].iloc[0]
            print(f"\n{col} Validation:")
            print(f"  Original: {sample_orig} ({type(sample_orig)})")
            print(f"  Transformed: {sample_opt} ({type(sample_opt)})")
            if isinstance(sample_opt, pd.Timestamp):
                try:
                    days_diff = (optimized_df[col].max() - optimized_df[col].min()).days
                    print(f"  Date range spans {days_diff} days")
                    validation_results[col] = "PASS"
                except Exception as e:
                    print(f"  Failed date operation: {str(e)}")
                    validation_results[col] = "FAIL"
    
    def check_numeric_columns():
        numeric_cols = ['strike', 'lastPrice', 'bid', 'ask', 'change', 
                        'percentChange', 'volume', 'openInterest', 'impliedVolatility']
        for col in numeric_cols:
            print(f"\n{col} Validation:")
            orig_stats = original_df[col].describe()
            opt_stats = optimized_df[col].describe()
            print(f"  Original range: {orig_stats['min']:.4f} to {orig_stats['max']:.4f}")
            print(f"  Optimized range: {opt_stats['min']:.4f} to {opt_stats['max']:.4f}")
            try:
                _ = optimized_df[col].mean() * 2
                _ = optimized_df[col].sum()
                print("  Calculations work.")
                validation_results[col] = "PASS"
            except Exception as e:
                print(f"  Calculation failed: {str(e)}")
                validation_results[col] = "FAIL"
    
    def check_categorical_columns():
        cat_cols = ['contractSize', 'currency']
        for col in cat_cols:
            print(f"\n{col} Validation:")
            orig_unique = set(original_df[col].unique())
            opt_unique = set(optimized_df[col].unique())
            print(f"  Original categories: {orig_unique}")
            print(f"  Optimized categories: {opt_unique}")
            if orig_unique == opt_unique:
                print("  Categories preserved.")
                validation_results[col] = "PASS"
            else:
                print("  WARNING: Categories changed during optimization")
                validation_results[col] = "FAIL"
    
    def check_option_specific_calculations():
        print("\nValidating Option-Specific Relationships:")
        try:
            call_itm = optimized_df[
                (optimized_df['contractSymbol'].str.contains('C')) & 
                (optimized_df['strike'] < optimized_df['stockClose'])
            ]
            put_itm = optimized_df[
                (optimized_df['contractSymbol'].str.contains('P')) & 
                (optimized_df['strike'] > optimized_df['stockClose'])
            ]
            print(f"  ITM calls found: {len(call_itm)}")
            print(f"  ITM puts found: {len(put_itm)}")
            optimized_df['spread'] = optimized_df['ask'] - optimized_df['bid']
            print(f"  Average bid-ask spread: {optimized_df['spread'].mean():.4f}")
            optimized_df['tte'] = (
                pd.to_datetime(optimized_df['expiryDate']) - pd.to_datetime(optimized_df['quoteDate'])
            ).dt.days
            print(f"  Average time-to-expiry (days): {optimized_df['tte'].mean():.1f}")
            validation_results['option_calculations'] = "PASS"
        except Exception as e:
            print(f"  Option calculations failed: {str(e)}")
            validation_results['option_calculations'] = "FAIL"
    
    check_date_columns()
    check_numeric_columns()
    check_categorical_columns()
    check_option_specific_calculations()
    
    print("\nValidation Summary:")
    print("=" * 80)
    for col, result in validation_results.items():
        print(f"  {col}: {result}")
    passed = all(result == "PASS" for result in validation_results.values())
    return passed

def test_optimization_strategy(sample_df):
    """
    Test the optimization strategy on sample data with NA handling.
    """
    print("\nTesting optimization strategy on sample data...")
    for col in sample_df.columns:
        na_count = sample_df[col].isna().sum()
        inf_count = np.isinf(pd.to_numeric(sample_df[col], errors='coerce')).sum()
        if na_count > 0 or inf_count > 0:
            print(f"  {col}: {na_count} NA values, {inf_count} inf values")
    try:
        optimized_df = sample_df.copy()
        # 1. Convert date columns
        date_cols = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_cols:
            optimized_df[col] = pd.to_datetime(optimized_df[col])
        # 2. Convert float columns (fill NA if needed)
        float32_cols = [
            'stockClose_ewm_5d', 'stockClose_ewm_15d', 'stockClose_ewm_45d',
            'stockClose_ewm_135d', 'stockHigh', 'stockLow', 'stockOpen',
            'stockClose', 'stockAdjClose', 'ask', 'bid', 'change',
            'strikeDelta', 'impliedVolatility'
        ]
        for col in float32_cols:
            if optimized_df[col].isna().any():
                fill_value = optimized_df[col].mean()
                optimized_df[col] = optimized_df[col].fillna(fill_value)
            optimized_df[col] = optimized_df[col].astype('float32')
        # 3. Convert integer columns
        int_cols = {'stockVolume': 'uint32', 'volume': 'uint32', 'openInterest': 'uint32'}
        for col, dtype in int_cols.items():
            if optimized_df[col].isna().any():
                optimized_df[col] = optimized_df[col].fillna(0)
            optimized_df[col] = optimized_df[col].clip(lower=0).astype(dtype)
        # 4. Convert categorical columns
        cat_cols = ['contractSize', 'currency']
        for col in cat_cols:
            if optimized_df[col].isna().any():
                fill_value = optimized_df[col].mode()[0]
                optimized_df[col] = optimized_df[col].fillna(fill_value)
            optimized_df[col] = optimized_df[col].astype('category')
        # 5. Clean daysToExpiry
        optimized_df['daysToExpiry'] = (optimized_df['daysToExpiry']
                                        .str.replace(' days', '')
                                        .fillna(0)
                                        .astype('uint16'))
        original_mem = sample_df.memory_usage(deep=True).sum() / 1024**2
        optimized_mem = optimized_df.memory_usage(deep=True).sum() / 1024**2
        savings = (1 - optimized_mem / original_mem) * 100
        print("\nOptimization Results:")
        print("=" * 50)
        print(f"  Original Memory: {original_mem:.2f} MB")
        print(f"  Optimized Memory: {optimized_mem:.2f} MB")
        print(f"  Memory Savings: {savings:.1f}%")
        for col in optimized_df.columns:
            print(f"\n{col}:")
            print(f"  Original dtype: {sample_df[col].dtype}")
            print(f"  Optimized dtype: {optimized_df[col].dtype}")
            print(f"  NA count before: {sample_df[col].isna().sum()}")
            print(f"  NA count after: {optimized_df[col].isna().sum()}")
        if validate_transformed_data(sample_df, optimized_df):
            print("\nAll validations passed – data transformation is safe!")
        else:
            print("\nWARNING: Some validations failed – review the details above.")
        return True, optimized_df
    except Exception as e:
        print(f"Error during optimization test: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False, None

# ==================== Main Pipeline ====================
def main():
    print("\n=== Data Processing Pipeline Start ===")
    
    # Step 1: Rearrange raw data
    rearrange_option_data()
    
    # Step 2: Clean the dataset
    clean_option_data()
    
    # Step 3: Analyze column completeness and quality
    analyze_columns()
    
    # Step 4: Test optimization strategy on a sample of the cleaned data
    sample_df = create_sample_data(CLEANED_CSV_FILE, sample_size=1000)
    success, optimized_sample = test_optimization_strategy(sample_df)
    if not success:
        print("\nSample optimization failed. Aborting full dataset processing.")
        return
    analyze_memory_usage(optimized_sample)
    
    # Step 5: Process the full dataset in chunks with datatype optimization
    process_full_dataset()
    
    print("\n=== Data Processing Pipeline Complete ===\n")

if __name__ == "__main__":
    main()