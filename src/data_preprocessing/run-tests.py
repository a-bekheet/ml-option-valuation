#!/usr/bin/env python3
"""
Test script for running validation tests on the sampled options data.
This script uses the previously developed OptionsDataTester class to run
comprehensive tests on the data preprocessing steps.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time

# Import the testing class (assuming it's in the same directory)
try:
    from options_preprocessing_tester import OptionsDataTester
except ImportError:
    print("Error: Could not import OptionsDataTester class. Make sure options_preprocessing_tester.py is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sampled_data_tests.log')
    ]
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run tests on sampled options data')
    
    parser.add_argument('--raw', default='sampled_data/option_data_with_headers_sample_99974rows_20250310_173210.csv',
                        help='Path to the raw sampled data file')
    parser.add_argument('--output-dir', default='test_results_sampled',
                        help='Directory to save test results')
    parser.add_argument('--ticker', default=None, 
                        help='Specific ticker to test (optional)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing steps and only run tests')
    
    return parser.parse_args()

def preprocess_sample(raw_file, processed_file):
    """
    Preprocess the sampled data file to prepare it for testing.
    This function applies:
    1. Data type adjustments
    2. Cyclical encoding of date features
    3. Normalization of numeric features
    """
    logging.info(f"Preprocessing sampled data: {raw_file}")
    
    try:
        # Load the data
        df = pd.read_csv(raw_file)
        logging.info(f"Loaded data with shape: {df.shape}")
        
        # 1. Adjust data types
        logging.info("Adjusting data types...")
        
        # Convert date columns
        date_cols = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                logging.info(f"Converted {col} to datetime")
        
        # Convert numeric columns
        numeric_cols = [
            'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange',
            'volume', 'openInterest', 'impliedVolatility', 'stockVolume',
            'stockClose', 'stockAdjClose', 'stockOpen', 'stockHigh', 'stockLow',
            'strikeDelta', 'stockClose_ewm_5d', 'stockClose_ewm_15d',
            'stockClose_ewm_45d', 'stockClose_ewm_135d'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logging.info(f"Converted {col} to numeric")
        
        # Handle days to expiry
        if 'daysToExpiry' in df.columns:
            if df['daysToExpiry'].dtype == object:
                df['daysToExpiry'] = df['daysToExpiry'].str.replace(' days', '').astype(float)
            df['daysToExpiry'] = df['daysToExpiry'].astype('int32')
            logging.info("Converted daysToExpiry to integer")
        
        # Convert inTheMoney to boolean
        if 'inTheMoney' in df.columns:
            df['inTheMoney'] = df['inTheMoney'].astype(bool)
            logging.info("Converted inTheMoney to boolean")
        
        # 2. Add cyclical encoding for date features
        logging.info("Adding cyclical encodings...")
        
        if 'lastTradeDate' in df.columns:
            # Day of week (Monday=0, Sunday=6)
            df['day_of_week'] = df['lastTradeDate'].dt.dayofweek
            
            # Day of month (1..31)
            df['day_of_month'] = df['lastTradeDate'].dt.day
            
            # Day of year (1..365 or 366)
            df['day_of_year'] = df['lastTradeDate'].dt.dayofyear
            
            # Cyclical encoding
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            df['day_of_month_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31)
            df['day_of_month_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31)
            
            df['day_of_year_sin'] = np.sin(2 * np.pi * (df['day_of_year'] - 1) / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)
            
            logging.info("Added cyclical temporal features")
        
        # 3. Normalize numeric features
        logging.info("Normalizing numeric features...")
        
        # Exclude cyclical and non-numeric columns
        exclude_cols = {
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'inTheMoney', 'ticker'
        }
        
        # Get columns to normalize
        normalize_cols = [col for col in numeric_cols if col in df.columns and col not in exclude_cols]
        
        # Store scaling parameters
        scaling_params = {}
        
        # Normalize each column
        for col in normalize_cols:
            values = df[col].dropna()
            if len(values) == 0:
                continue
                
            mean = values.mean()
            std = values.std()
            
            # Skip if std is 0
            if std == 0:
                logging.warning(f"Column {col} has zero standard deviation, skipping normalization")
                continue
            
            # Store parameters
            scaling_params[col] = {'mean': float(mean), 'std': float(std)}
            
            # Apply normalization
            df.loc[df[col].notna(), col] = (df.loc[df[col].notna(), col] - mean) / std
            logging.info(f"Normalized {col}: mean={mean:.4f}, std={std:.4f}")
        
        # Save scaling parameters
        params_dir = os.path.dirname(processed_file)
        os.makedirs(params_dir, exist_ok=True)
        
        params_file = os.path.join(params_dir, 'scaling_params.json')
        with open(params_file, 'w') as f:
            json.dump(scaling_params, f, indent=2)
        logging.info(f"Saved scaling parameters to {params_file}")
        
        # Save processed data
        df.to_csv(processed_file, index=False)
        logging.info(f"Saved processed data to {processed_file}")
        
        return processed_file, params_file
        
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def main():
    """Main function to run tests on sampled data."""
    args = parse_arguments()
    
    # Set up output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define processed file path
    processed_file = os.path.join(args.output_dir, 'processed_sample.csv')
    
    # Preprocess data if needed
    if not args.skip_preprocessing:
        processed_file, scaling_params_file = preprocess_sample(args.raw, processed_file)
        if not processed_file:
            logging.error("Preprocessing failed. Exiting.")
            return
    else:
        # If skipping preprocessing, check if processed file exists
        if not os.path.exists(processed_file):
            logging.error(f"Processed file not found: {processed_file}")
            logging.error("Please run without --skip-preprocessing or provide the correct path.")
            return
        
        # Look for scaling parameters
        scaling_params_file = os.path.join(args.output_dir, 'scaling_params.json')
        if not os.path.exists(scaling_params_file):
            logging.warning(f"Scaling parameters file not found: {scaling_params_file}")
            scaling_params_file = None
    
    # Initialize tester
    tester = OptionsDataTester(output_dir=args.output_dir)
    
    # Run tests
    start_time = time.time()
    logging.info("Running tests on processed data...")
    
    test_results = tester.run_all_tests(
        raw_file=args.raw,
        processed_file=processed_file,
        scaling_params_file=scaling_params_file,
        specific_ticker=args.ticker
    )
    
    # Print summary
    print("\nTest Results Summary:")
    print("=" * 50)
    
    if 'error' in test_results:
        print(f"Tests failed with error: {test_results['error']}")
        return
    
    print(f"Overall Success: {'✅' if test_results.get('overall_success', False) else '❌'}")
    
    normalization_success = test_results.get('normalization', {}).get('all_normalized', False)
    print(f"Normalization: {'✅' if normalization_success else '❌'}")
    
    recovery_success = test_results.get('recovery', {}).get('all_recoverable', False)
    print(f"Data Recovery: {'✅' if recovery_success else '❌'}")
    
    cyclical_success = test_results.get('cyclical', {}).get('all_valid', False)
    print(f"Cyclical Features: {'✅' if cyclical_success else '❌'}")
    
    total_time = time.time() - start_time
    print(f"\nTests completed in {total_time:.2f} seconds")
    print(f"Test results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()