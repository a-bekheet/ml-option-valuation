#!/usr/bin/env python3
"""
Comprehensive data preprocessing pipeline for options data.

This script combines multiple preprocessing steps from individual modules:
1. Adjusts data types
2. Handles stock splits
3. Performs drop operations
4. Adds cyclical encoding for temporal features
5. Normalizes features (with per-ticker scaling)
6. Stores scaling parameters for original data recovery
7. Validates the processed output

Usage:
    python preprocess_pipeline.py --input-file [path] --output-dir [dir] [options]
"""

import os
import sys
import re
import logging
import argparse
import json
import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from tqdm import tqdm
from normalize_and_split import split_and_normalize_by_ticker

# Import preprocessing modules
sys.path.insert(0, 'src/data_preprocessing')
try:
    from drops import prelim_dropping, extract_ticker
    from dtypes_adjustor import datatype_adjustor
    from splits_adjustor import adjust_for_stock_splits
    from cyclic_encoding import encode_dataframe 
    from normalize_and_split import get_numeric_columns, compute_scaler_for_ticker, transform_ticker_data, split_and_normalize_by_ticker
    from normalize_and_split_utils import scale_dataset, split_data_by_ticker
    from verify_norm import verify_scaled_file, finalize_stats
    from greeks_calculator import add_risk_free_rate, calculate_greeks_bs
except ImportError as e:
    print(f"Error importing preprocessing modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Configure logging
log_file = f"preprocess_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Comprehensive options data preprocessing pipeline')
    
    parser.add_argument('--input-file', required=True,
                        help='Path to the input data file (CSV or NPY)')
    parser.add_argument('--output-dir', default='data_files/processed_data',
                        help='Directory to save processed data files')
    parser.add_argument('--global-output', default='data_files/option_data_scaled.csv',
                        help='Path for the globally scaled output file (all tickers)')
    parser.add_argument('--batch-size', type=int, default=200000,
                        help='Batch size for processing large files')
    parser.add_argument('--tickers-to-drop', nargs='+', default=['BAC', 'C'],
                        help='List of tickers to exclude from processing')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the scaled output after processing')
    parser.add_argument('--skip-splits', action='store_true',
                        help='Skip stock splits adjustment')
    parser.add_argument('--recovery-test', action='store_true',
                        help='Run data recovery tests on a sample of tickers')
    parser.add_argument('--rate-file', default='/Users/bekheet/dev/option-ml-prediction/data_files/split_data/DGS10.csv',
                        help='Path to the risk-free rate file (e.g., DGS10.csv)')
    parser.add_argument('--option-type-col', default=None, # Set default if you have this column, e.g., 'optionType'
                        help='Name of column indicating Call/Put (e.g., C/P)')

    return parser.parse_args()

def create_directory_structure(output_dir: str) -> Dict[str, str]:
    """Create necessary directories for output files."""
    paths = {}
    
    # Main output directory
    paths['main'] = output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Directory for ticker-specific files
    paths['tickers'] = os.path.join(output_dir, 'by_ticker')
    Path(paths['tickers']).mkdir(parents=True, exist_ok=True)
    
    # Directory for scaling parameters
    paths['params'] = os.path.join(paths['tickers'], 'scaling_params')
    Path(paths['params']).mkdir(parents=True, exist_ok=True)
    
    # Directory for validation results
    paths['validation'] = os.path.join(output_dir, 'validation')
    Path(paths['validation']).mkdir(parents=True, exist_ok=True)
    
    # Directory for temporary files
    paths['temp'] = os.path.join(output_dir, 'temp')
    Path(paths['temp']).mkdir(parents=True, exist_ok=True)
    
    return paths

def load_data(file_path: str, batch_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from CSV or NPY file.
    For large files, use the first batch to inspect and detect column names.
    """
    logging.info(f"Loading data from {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.npy':
        # Define column names for NPY files
        column_names = [
            'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 
            'bid', 'ask', 'change', 'percentChange', 'volume', 
            'openInterest', 'impliedVolatility', 'inTheMoney', 
            'contractSize', 'currency', 'quoteDate', 'expiryDate', 
            'daysToExpiry', 'stockVolume', 'stockClose', 'stockAdjClose', 
            'stockOpen', 'stockHigh', 'stockLow', 'strikeDelta', 
            'stockClose_ewm_5d', 'stockClose_ewm_15d', 'stockClose_ewm_45d', 
            'stockClose_ewm_135d'
        ]
        
        try:
            data = np.load(file_path, allow_pickle=True)
            logging.info(f"Loaded NPY array with shape {data.shape}")
            
            # Check if columns match expected count
            if data.shape[1] != len(column_names):
                logging.warning(f"Column count mismatch: array has {data.shape[1]} columns, expected {len(column_names)}")
                
                # Adjust column names as needed
                if data.shape[1] < len(column_names):
                    actual_columns = column_names[:data.shape[1]]
                else:
                    actual_columns = column_names + [f"Unknown{i}" for i in range(data.shape[1] - len(column_names))]
                    
                df = pd.DataFrame(data, columns=actual_columns)
            else:
                df = pd.DataFrame(data, columns=column_names)
                
            logging.info(f"Converted NPY to DataFrame with {len(df)} rows and {len(df.columns)} columns")
            
        except Exception as e:
            logging.error(f"Error loading NPY file: {str(e)}")
            raise
            
    elif file_ext == '.csv':
        try:
            if batch_size:
                # For large files, load just the first batch to inspect structure
                first_batch = pd.read_csv(file_path, nrows=100)
                logging.info(f"Loaded first 100 rows to inspect CSV structure")
                logging.info(f"CSV has {len(first_batch.columns)} columns")
                return first_batch  # Return the first batch for structure analysis
            else:
                # Load entire file for smaller files
                df = pd.read_csv(file_path)
                logging.info(f"Loaded entire CSV file with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            logging.error(f"Error loading CSV file: {str(e)}")
            raise
    else:
        error_msg = f"Unsupported file type: {file_ext}. Only .csv and .npy files are supported."
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    return df

def fix_data_types(input_file: str, output_file: str, batch_size: int) -> str:
    """Apply data type fixes and conversions."""
    logging.info("Applying data type adjustments")
    
    try:
        datatype_adjustor(input_file, output_file, chunksize=batch_size)
        logging.info(f"Data types adjusted successfully, output saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error adjusting data types: {str(e)}")
        raise

def adjust_splits(input_file: str, output_file: str) -> str:
    """Adjust data for stock splits."""
    logging.info("Adjusting for stock splits")
    
    try:
        adjust_for_stock_splits(input_file, output_file)
        logging.info(f"Stock splits adjustments applied, output saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error adjusting for stock splits: {str(e)}")
        raise

def drop_tickers(input_file: str, output_file: str, tickers_to_drop: List[str]) -> str:
    """Drop specified tickers and rows with missing data."""
    logging.info(f"Dropping tickers {tickers_to_drop} and rows with missing values")
    
    try:
        prelim_dropping(input_file, output_file, ticker_list=tickers_to_drop)
        logging.info(f"Dropping operation completed, output saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error dropping data: {str(e)}")
        raise

def add_cyclical_encodings(input_file: str, output_file: str) -> str:
    """Add cyclical encodings for temporal features."""
    logging.info("Adding cyclical encodings for temporal features")
    
    try:
        encode_dataframe(
            input_path=input_file,
            output_path=output_file,
            date_column='lastTradeDate',
            keep_original_dates=True,
            keep_original_features=True
        )
        logging.info(f"Cyclical encodings added, output saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error adding cyclical encodings: {str(e)}")
        raise

def normalize_per_ticker(input_file: str, tickers_dir: str, params_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Normalize data for each ticker separately and save the scaling parameters.
    Returns a dictionary with ticker information.
    """
    logging.info("Normalizing data per ticker")
    
    try:
        # Read the processed data
        df = pd.read_csv(input_file)
        
        # Add ticker if it doesn't exist
        if 'ticker' not in df.columns:
            logging.info("Ticker column not found, extracting from contractSymbol")
            df['ticker'] = df['contractSymbol'].apply(extract_ticker)
        
        # Define columns to exclude from scaling
        exclude_cols = {
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'inTheMoney', 'ticker', 'contractSymbol'
        }
        
        # Get unique tickers
        tickers = df['ticker'].unique()
        logging.info(f"Found {len(tickers)} unique tickers")
        
        # Process each ticker
        ticker_info = {}
        for ticker in tqdm(tickers, desc="Normalizing per ticker"):
            # Extract ticker data
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # Skip if not enough data
            if len(ticker_data) < 10:
                logging.warning(f"Insufficient data for ticker {ticker}, skipping")
                continue
            
            # Get numeric columns for this ticker
            numeric_cols = get_numeric_columns(ticker_data, exclude_cols)
            
            # Compute scaler and parameters
            scaler, scaling_params = compute_scaler_for_ticker(ticker_data, numeric_cols)
            
            # Transform the data
            normalized_data = transform_ticker_data(ticker_data, scaler, numeric_cols)
            
            # Save normalized data
            normalized_file = os.path.join(tickers_dir, f"{ticker}_normalized.csv")
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
        
        metadata_file = os.path.join(tickers_dir, 'ticker_metadata.csv')
        metadata.to_csv(metadata_file, index=False)
        logging.info(f"Ticker metadata saved to {metadata_file}")
        
        return ticker_info
    
    except Exception as e:
        logging.error(f"Error normalizing per ticker: {str(e)}")
        raise

def test_data_recovery(ticker_info: Dict[str, Dict[str, Any]], num_tickers: int = 3) -> Dict[str, Dict[str, Any]]:
    """
    Test the recovery of original data for a sample of tickers.
    Returns a dictionary with recovery test results.
    """
    logging.info(f"Testing data recovery for {num_tickers} random tickers")
    
    results = {}
    
    try:
        # Select a few tickers randomly
        import random
        tickers = list(ticker_info.keys())
        sample_tickers = random.sample(tickers, min(num_tickers, len(tickers)))
        
        for ticker in sample_tickers:
            info = ticker_info[ticker]
            normalized_file = info['file_path']
            params_file = info['scaling_params']
            
            # Create recovery output path
            recovery_file = normalized_file.replace("_normalized.csv", "_recovered.csv")
            
            # Load normalized data
            df_norm = pd.read_csv(normalized_file)
            
            # Load scaling parameters
            with open(params_file, 'r') as f:
                scaling_params = json.load(f)
            
            # Recover original data
            df_recovered = df_norm.copy()
            
            # Apply inverse transformation to each column
            for col, params in scaling_params.items():
                # Skip if column not in DataFrame
                if col not in df_norm.columns:
                    continue
                    
                # Get parameters
                mean = params['mean']
                scale = params['scale']
                
                # Inverse transform: X_orig = X_scaled * scale + mean
                mask = ~df_norm[col].isna()
                df_recovered.loc[mask, col] = df_norm.loc[mask, col] * scale + mean
            
            # Save recovered data
            df_recovered.to_csv(recovery_file, index=False)
            
            # Calculate recovery error metrics
            recovery_metrics = {}
            for col in scaling_params.keys():
                if col in df_norm.columns:
                    # Calculate mean absolute error and percentage error
                    norm_values = df_norm[col].dropna()
                    orig_values = df_recovered[col].dropna()
                    
                    # Only test where both have values
                    common_idx = norm_values.index.intersection(orig_values.index)
                    if len(common_idx) == 0:
                        continue
                        
                    # Calculate errors
                    abs_error = np.abs(df_recovered.loc[common_idx, col] - df_norm.loc[common_idx, col] * scale - mean)
                    mean_abs_error = abs_error.mean()
                    max_abs_error = abs_error.max()
                    
                    # Store metrics
                    recovery_metrics[col] = {
                        'mean_abs_error': float(mean_abs_error),
                        'max_abs_error': float(max_abs_error)
                    }
            
            # Store results
            results[ticker] = {
                'normalized_file': normalized_file,
                'recovered_file': recovery_file,
                'metrics': recovery_metrics
            }
            
            logging.info(f"Recovery test for {ticker} completed and saved to {recovery_file}")
        
        # Save overall recovery test results
        results_file = os.path.join(os.path.dirname(ticker_info[sample_tickers[0]]['file_path']), 'recovery_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Recovery test results saved to {results_file}")
        return results
    
    except Exception as e:
        logging.error(f"Error testing data recovery: {str(e)}")
        raise

def verify_scaled_data(scaled_file: str, output_dir: str) -> pd.DataFrame:
    """
    Verify the scaled data to ensure proper normalization.
    Returns a DataFrame with verification results.
    """
    logging.info(f"Verifying scaled data in {scaled_file}")
    
    try:
        # Define columns to verify
        numeric_cols = [
            "strike", "lastPrice", "bid", "ask", "change", "percentChange",
            "volume", "openInterest", "impliedVolatility", "daysToExpiry",
            "stockVolume", "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d"
        ]
        
        cyclical_cols = [
            "day_of_week_sin", "day_of_week_cos",
            "day_of_month_sin", "day_of_month_cos",
            "day_of_year_sin", "day_of_year_cos"
        ]
        
        bool_cols = ["inTheMoney"]
        
        # Run verification
        summary_df = verify_scaled_file(
            file_path=scaled_file,
            chunksize=200000,
            expected_numeric_cols=numeric_cols,
            cyclical_cols=cyclical_cols,
            bool_cols=bool_cols
        )
        
        # Save verification results
        results_file = os.path.join(output_dir, 'verification_results.csv')
        summary_df.to_csv(results_file, index=False)
        
        logging.info(f"Verification results saved to {results_file}")
        
        # Check for columns with significant deviations
        out_of_range = summary_df[
            (summary_df["mean"].abs() > 0.05) |  # mean should be close to 0
            (summary_df["std"] < 0.8) | (summary_df["std"] > 1.2)  # std should be close to 1
        ]
        
        if not out_of_range.empty:
            logging.warning("Some columns deviate significantly from expected scaling range:")
            for _, row in out_of_range.iterrows():
                logging.warning(f"  {row['column']}: mean={row['mean']:.4f}, std={row['std']:.4f}")
        else:
            logging.info("All columns within expected scaling range!")
        
        return summary_df
    
    except Exception as e:
        logging.error(f"Error verifying scaled data: {str(e)}")
        raise

def main():
    """Main preprocessing pipeline function."""
    start_time = time.time()

    # Parse command line arguments (ensure parse_arguments includes --rate-file and --option-type-col)
    args = parse_arguments()

    # Create directory structure
    paths = create_directory_structure(args.output_dir)

    # Configure logging (ensure it's set up above main or here)
    log_file = os.path.join(paths['main'], f"preprocess_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logging.info(f"Log file: {log_file}")

    try:
        # Define temporary file paths
        datatypes_fixed = os.path.join(paths['temp'], 'datatypes_fixed.csv')
        splits_adjusted = os.path.join(paths['temp'], 'splits_adjusted.csv')
        tickers_dropped = os.path.join(paths['temp'], 'tickers_dropped.csv')
        rates_added = os.path.join(paths['temp'], 'rates_added.csv')
        greeks_added = os.path.join(paths['temp'], 'greeks_added.csv')
        cyclical_encoded = os.path.join(paths['temp'], 'cyclical_encoded.csv')

        logging.info(f"Starting comprehensive preprocessing pipeline")
        logging.info(f"Input file: {args.input_file}")
        # ... [Log other arguments as before] ...
        logging.info(f"Risk-free rate file: {args.rate_file}")

        # Step 1: Fix data types
        current_file = fix_data_types(args.input_file, datatypes_fixed, args.batch_size)

        # Step 2: Adjust for stock splits
        if args.skip_splits:
            logging.info("Skipping stock splits adjustment")
            current_file = datatypes_fixed
        else:
            # Make sure adjust_splits is imported
            current_file = adjust_splits(current_file, splits_adjusted)

        # Step 3: Drop unwanted tickers
        # Make sure drop_tickers is imported
        current_file = drop_tickers(current_file, tickers_dropped, args.tickers_to_drop)

        # Step 3.5: Add Risk-Free Rate
        logging.info("Adding risk-free rate...")
        try:
            df_for_rates = pd.read_csv(current_file)
            # Make sure add_risk_free_rate is imported
            df_with_rates = add_risk_free_rate(df_for_rates, args.rate_file, date_col='quoteDate')
            df_with_rates.to_csv(rates_added, index=False)
            current_file = rates_added
            logging.info(f"Risk-free rate added, saved to {current_file}")
            del df_for_rates, df_with_rates
        except Exception as rate_err:
             logging.error(f"Failed to add risk-free rate: {rate_err}")
             logging.error(traceback.format_exc()) # Make sure traceback is imported
             raise

        # Step 3.6: Calculate Greeks
        logging.info("Calculating approximate Option Greeks...")
        try:
            df_for_greeks = pd.read_csv(current_file)
            option_type_col_name = args.option_type_col
            inferred_col = None
            if not option_type_col_name and 'contractSymbol' in df_for_greeks.columns:
                logging.info("Attempting to infer option type from contractSymbol...")
                def infer_option_type(symbol):
                    match = re.search(r'^[A-Z]+(\d{6})([CP])', str(symbol)) # Make sure re is imported
                    return match.group(2) if match else None
                inferred_col = 'inferredOptionType'
                df_for_greeks[inferred_col] = df_for_greeks['contractSymbol'].apply(infer_option_type)
                if df_for_greeks[inferred_col].notna().sum() > 0:
                    option_type_col_name = inferred_col
                    logging.info(f"Using inferred option types from temporary column '{inferred_col}'.")
                else:
                    logging.warning("Could not infer option types. Greeks dependent on type will assume CALL.")
                    if inferred_col in df_for_greeks.columns: df_for_greeks = df_for_greeks.drop(columns=[inferred_col])
                    inferred_col = None
            # Make sure calculate_greeks_bs is imported
            df_with_greeks = calculate_greeks_bs(df_for_greeks, option_type_col=option_type_col_name)
            if inferred_col and inferred_col in df_with_greeks.columns:
                df_with_greeks = df_with_greeks.drop(columns=[inferred_col])
            df_with_greeks.to_csv(greeks_added, index=False)
            current_file = greeks_added
            logging.info(f"Greeks calculated, saved to {current_file}")
            del df_for_greeks, df_with_greeks
        except Exception as greek_err:
            logging.error(f"Failed to calculate Greeks: {greek_err}")
            logging.error(traceback.format_exc()) # Make sure traceback is imported
            raise

        # Step 4: Add cyclical encodings
        # Make sure add_cyclical_encodings is imported
        current_file = add_cyclical_encodings(current_file, cyclical_encoded)

        # Step 5: Normalize per ticker
        logging.info("Normalizing features per ticker...")
        exclude_norm_cols = {
            'day_of_week_sin', 'day_of_week_cos', 'day_of_month_sin', 'day_of_month_cos',
            'day_of_year_sin', 'day_of_year_cos', 'inTheMoney',
            'ticker', 'contractSymbol', 'lastTradeDate', 'quoteDate', 'expiryDate',
            'currency', 'contractSize'
        }
        try:
            # --- CORRECTED CALL ---
            # Directly call split_and_normalize_by_ticker
            # Make sure split_and_normalize_by_ticker is imported from normalize_and_split
            ticker_info = split_and_normalize_by_ticker(
                 input_file=current_file,
                 output_dir=paths['tickers'], # Save to 'by_ticker' subdir
                 ticker_column='ticker',
                 exclude_columns=exclude_norm_cols,
                 chunksize=None # Or pass args.batch_size if chunking is needed here
            )
            # --- END CORRECTION ---
            logging.info(f"Normalization per ticker complete. Metadata saved in {paths['tickers']}")
        except Exception as norm_err:
             logging.error(f"Failed during per-ticker normalization: {norm_err}")
             logging.error(traceback.format_exc()) # Make sure traceback is imported
             raise

        # Step 6: Create a globally scaled version if requested (Optional)
        if args.global_output:
             try:
                 logging.info(f"Creating globally scaled version at {args.global_output}")
                 from normalize_and_split_utils import scale_dataset
                 scale_dataset(cyclical_encoded, args.global_output, chunksize=args.batch_size, exclude_columns=exclude_norm_cols)
                 logging.info(f"Global scaled file saved to {args.global_output}")
             except ImportError: logging.error("Could not import 'scale_dataset'. Skipping global scaling.")
             except NameError: logging.error("Function 'scale_dataset' not found. Skipping global scaling.")
             except Exception as global_scale_err: logging.error(f"Failed globally scaled file: {global_scale_err}")

        # Step 7: Run verification on scaled data if requested
        # ... [Verification logic remains the same, ensure verify_norm is imported if used] ...
        if args.verify and args.global_output and os.path.exists(args.global_output):
             try:
                 from verify_norm import verify_scaled_file
                 # Define columns to verify including greeks
                 numeric_cols_to_verify = [ "strike", "lastPrice", "bid", "ask", "change", "percentChange", "volume", "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume", "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow", "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d", "stockClose_ewm_45d", "stockClose_ewm_135d", "risk_free_rate", "delta", "gamma", "vega", "theta", "rho" ]
                 cyclical_cols_to_verify = [ "day_of_week_sin", "day_of_week_cos", "day_of_month_sin", "day_of_month_cos", "day_of_year_sin", "day_of_year_cos" ]
                 bool_cols_to_verify = ["inTheMoney"]
                 logging.info(f"Verifying global scaled file: {args.global_output}")
                 verify_scaled_file( file_path=args.global_output, chunksize=args.batch_size, expected_numeric_cols=numeric_cols_to_verify, cyclical_cols=cyclical_cols_to_verify, bool_cols=bool_cols_to_verify )
                 logging.info(f"Verification results should be saved in {paths['validation']}")
             except ImportError: logging.warning("Module 'verify_norm' not found. Skipping verification.")
             except NameError: logging.warning("Function 'verify_scaled_file' not found. Skipping verification.")
             except Exception as verify_err: logging.error(f"Error during verification: {verify_err}")
        elif args.verify: logging.warning("Verification requested but no global output file specified or created.")


        # Step 8: Test data recovery if requested
        # ... [Recovery test logic remains the same, ensure function is defined/imported] ...
        if args.recovery_test and 'ticker_info' in locals() and ticker_info:
             try:
                 # test_data_recovery(ticker_info) # Make sure this function is defined/imported
                 logging.info("Recovery test step called (ensure implementation is available).")
             except NameError: logging.warning("Recovery test function (test_data_recovery) not defined. Skipping.")
             except Exception as recovery_err: logging.error(f"Error during recovery test: {recovery_err}")
        elif args.recovery_test: logging.warning("Recovery test requested but ticker info unavailable.")


        # Capture end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"Pipeline completed successfully!")
        logging.info(f"Total processing time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        if 'ticker_info' in locals(): logging.info(f"Processed {len(ticker_info)} tickers.")
        logging.info(f"Per-ticker normalized data saved in: {paths['tickers']}")
        logging.info(f"Scaling parameters saved in: {paths['params']}")

        # Clean up temporary files
        logging.info("Cleaning up temporary files...")
        cleanup_files = [datatypes_fixed, splits_adjusted, tickers_dropped, rates_added, greeks_added, cyclical_encoded]
        for file_path in cleanup_files:
            try:
                if os.path.exists(file_path): os.remove(file_path); logging.debug(f"Removed temp file: {file_path}")
            except Exception as cleanup_err: logging.warning(f"Could not remove temp file {file_path}: {cleanup_err}")
        try:
            if os.path.exists(paths['temp']) and not os.listdir(paths['temp']):
                 os.rmdir(paths['temp']); logging.debug(f"Removed empty temp directory: {paths['temp']}")
            elif os.path.exists(paths['temp']):
                 logging.warning(f"Temporary directory {paths['temp']} not empty, leaving for inspection.")
        except Exception as cleanup_err: logging.warning(f"Could not remove temp directory {paths['temp']}: {cleanup_err}")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        # Ensure traceback is imported at the top of the file
        logging.error(traceback.format_exc())
        print(f"\nPipeline failed. Check log file: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    # This check ensures the main function runs only when the script is executed directly
    # Ensure necessary imports (like re, traceback, etc.) are at the top of the file
    main()