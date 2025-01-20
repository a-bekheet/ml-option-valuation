import pandas as pd
import numpy as np
import time
from datetime import datetime
import psutil
import os
import warnings
warnings.filterwarnings('ignore')

# Add memory usage tracking at start
process = psutil.Process()
initial_memory = process.memory_info().rss / 1024**2
print(f"Initial memory usage: {initial_memory:.2f} MB")

def process_full_dataset(input_file, output_file, chunksize=100000):
    """
    Process the full dataset in chunks with tracking and validation checkpoints
    """
    # Get total rows for progress tracking
    total_rows = sum(1 for _ in open(input_file)) - 1
    print(f"Total rows to process: {total_rows:,}")
    
    # Read first chunk to understand data types
    sample_df = pd.read_csv(input_file, nrows=5)
    print("\nInitial column dtypes:")
    for col in sample_df.columns:
        print(f"{col}: {sample_df[col].dtype}")
    
    # Define initial dtypes for reading
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
        # Read these as float64 first, then convert
        'stockVolume': 'float64',
        'volume': 'float64',
        'openInterest': 'float64'
    }
    
    # Date columns for parsing
    date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
    
    def convert_types(chunk):
        """Convert datatypes after reading"""
        # Convert float columns to integers where appropriate
        int_conversions = {
            'stockVolume': 'uint32',
            'volume': 'uint32',
            'openInterest': 'uint32'
        }
        
        for col, dtype in int_conversions.items():
            # Fill NA values with 0 and convert to integer
            chunk[col] = chunk[col].fillna(0).astype(dtype)
        
        # Convert date columns
        for col in date_columns:
            chunk[col] = pd.to_datetime(chunk[col])
        
        # Convert daysToExpiry
        chunk['daysToExpiry'] = (chunk['daysToExpiry']
                                .str.replace(' days', '')
                                .fillna(0)
                                .astype('uint16'))
        
        return chunk

    # Initialize tracking variables
    rows_processed = 0
    chunks_processed = 0
    validation_frequency = 10
    
    try:
        start_time = time.time()
        first_chunk = True
        
        print(f"\nStarting processing at {datetime.now()}")
        
        # Process chunks
        for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize, dtype=dtypes), 1):
            chunk_start_time = time.time()
            
            # Convert datatypes
            chunk = convert_types(chunk)
            
            # Write chunk
            chunk.to_csv(output_file, 
                        mode='w' if first_chunk else 'a',
                        header=first_chunk,
                        index=False)
            
            if first_chunk:
                first_chunk = False
                print("\nFirst chunk dtypes after conversion:")
                for col in chunk.columns:
                    print(f"{col}: {chunk[col].dtype}")
            
            # Update tracking
            rows_processed += len(chunk)
            chunks_processed += 1
            
            # Print progress
            if chunks_processed % 5 == 0:  # Print every 5 chunks
                percent_complete = (rows_processed / total_rows) * 100
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (rows_processed / total_rows)
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"\nProgress Update:")
                print(f"Processed: {rows_processed:,}/{total_rows:,} rows ({percent_complete:.1f}%)")
                print(f"Memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
                print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
                print(f"Estimated time remaining: {remaining_time/60:.1f} minutes")
        
        # Final statistics
        print("\nProcessing Complete!")
        print("=" * 50)
        total_time = time.time() - start_time
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Final row count: {rows_processed:,}")
        print(f"Final memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    input_file = 'data_files/option_data_with_headers_cleaned.csv'
    output_file = 'data_files/option_data_compressed.csv'
    
    # Verify input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        process_full_dataset(input_file, output_file)
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        import traceback
        print(traceback.format_exc())