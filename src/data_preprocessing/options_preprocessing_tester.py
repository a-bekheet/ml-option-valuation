import numpy as np
import pandas as pd
import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import time
import argparse

# Import preprocessing components
try:
    from drops import Drop
    from dtypes_adjustor import datatype_adjustor
    from splits_adjustor import adjust_for_stock_splits
    from cyclic_encoding import encode_dataframe
    from normalize_and_split import get_numeric_columns, compute_scaler_for_ticker, transform_ticker_data
    from normalize_and_split_utils import scale_dataset, split_data_by_ticker
    from preprocess_data import preprocess_data
except ImportError as e:
    print(f"Warning: Could not import preprocessing modules: {e}")
    print("Continuing with test functionality only.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing_test.log')
    ]
)

class OptionsDataTester:
    """Test harness for options data preprocessing pipeline"""
    
    def __init__(self, output_dir: str = 'test_results'):
        """
        Initialize the tester.
        
        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Expected columns after preprocessing
        self.expected_numeric_cols = [
            'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange',
            'volume', 'openInterest', 'impliedVolatility', 'daysToExpiry', 
            'stockVolume', 'stockClose', 'stockAdjClose', 'stockOpen', 
            'stockHigh', 'stockLow', 'strikeDelta'
        ]
        
        self.expected_date_cols = [
            'lastTradeDate', 'quoteDate', 'expiryDate'
        ]
        
        self.expected_cyclical_cols = [
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'day_of_week', 'day_of_month', 'day_of_year'
        ]
    
    def load_test_data(self, 
                       raw_file: str, 
                       processed_file: str, 
                       ticker: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw and processed data files for testing.
        
        Args:
            raw_file: Path to original data file
            processed_file: Path to processed data file
            ticker: Optional ticker to filter data (for ticker-specific testing)
            
        Returns:
            Tuple of (raw_df, processed_df)
        """
        logging.info(f"Loading data files for testing:")
        logging.info(f"  Raw file: {raw_file}")
        logging.info(f"  Processed file: {processed_file}")
        
        # Load raw data
        if raw_file.endswith('.npy'):
            # Define column names for .npy file
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
            
            raw_data = np.load(raw_file, allow_pickle=True)
            raw_df = pd.DataFrame(raw_data, columns=column_names)
            logging.info(f"Loaded raw .npy file, shape: {raw_df.shape}")
        else:
            raw_df = pd.read_csv(raw_file)
            logging.info(f"Loaded raw CSV file, shape: {raw_df.shape}")
        
        # Load processed data
        processed_df = pd.read_csv(processed_file)
        logging.info(f"Loaded processed CSV file, shape: {processed_df.shape}")
        
        # Filter by ticker if specified
        if ticker:
            if 'ticker' not in raw_df.columns:
                # Extract ticker from contractSymbol
                raw_df['ticker'] = raw_df['contractSymbol'].str.extract(r'([A-Z]+)')
                logging.info("Extracted ticker from contractSymbol in raw data")
            
            raw_df = raw_df[raw_df['ticker'] == ticker]
            processed_df = processed_df[processed_df['ticker'] == ticker]
            
            logging.info(f"Filtered to ticker {ticker}:")
            logging.info(f"  Raw data shape: {raw_df.shape}")
            logging.info(f"  Processed data shape: {processed_df.shape}")
        
        return raw_df, processed_df
    
    def test_normalization(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test if numeric columns are properly normalized.
        
        Args:
            processed_df: DataFrame with processed data
            
        Returns:
            Dictionary with test results
        """
        logging.info("Testing normalization of numeric columns")
        
        # Identify numeric columns
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        
        # Exclude cyclical features which should be in [-1, 1]
        cyclical_cols = [col for col in self.expected_cyclical_cols if col in processed_df.columns]
        cols_to_test = [col for col in numeric_cols if col not in cyclical_cols and col != 'inTheMoney']
        
        # Test statistics for normalized columns
        stats = {
            'column': [],
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'within_expected_range': []
        }
        
        for col in cols_to_test:
            series = processed_df[col].dropna()
            if len(series) == 0:
                logging.warning(f"Column {col} has no non-null values, skipping")
                continue
                
            mean = series.mean()
            std = series.std()
            min_val = series.min()
            max_val = series.max()
            
            # For standard scaling, expect mean ≈ 0, std ≈ 1
            is_normalized = abs(mean) < 0.1 and 0.9 < std < 1.1
            
            stats['column'].append(col)
            stats['mean'].append(mean)
            stats['std'].append(std)
            stats['min'].append(min_val)
            stats['max'].append(max_val)
            stats['within_expected_range'].append(is_normalized)
            
            logging.info(f"Column {col}: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}, normalized={is_normalized}")
        
        # Test cyclical columns
        for col in cyclical_cols:
            if col in processed_df.columns:
                series = processed_df[col].dropna()
                if len(series) == 0:
                    continue
                    
                min_val = series.min()
                max_val = series.max()
                
                # For sin/cos columns, expect range [-1, 1]
                if '_sin' in col or '_cos' in col:
                    is_valid = min_val >= -1.01 and max_val <= 1.01
                    
                    stats['column'].append(col)
                    stats['mean'].append(series.mean())
                    stats['std'].append(series.std())
                    stats['min'].append(min_val)
                    stats['max'].append(max_val)
                    stats['within_expected_range'].append(is_valid)
                    
                    logging.info(f"Cyclical column {col}: min={min_val:.4f}, max={max_val:.4f}, valid range={is_valid}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(stats)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'normalization_test.csv')
        results_df.to_csv(results_file, index=False)
        logging.info(f"Saved normalization test results to {results_file}")
        
        # Visualize distribution of means and stds
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['mean'], results_df['std'], alpha=0.7)
        plt.axhline(y=1, color='r', linestyle='--')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlim(-0.5, 0.5)
        plt.ylim(0, 2)
        for i, col in enumerate(results_df['column']):
            plt.annotate(col, (results_df['mean'][i], results_df['std'][i]))
        plt.title('Normalization Test: Mean vs Std Dev')
        plt.xlabel('Mean (should be close to 0)')
        plt.ylabel('Std Dev (should be close to 1)')
        plt.grid(True, alpha=0.3)
        
        plot_file = os.path.join(self.output_dir, 'normalization_plot.png')
        plt.savefig(plot_file)
        logging.info(f"Saved normalization plot to {plot_file}")
        
        # Check overall result
        all_normalized = all(results_df[results_df['column'].str.contains('_sin|_cos') == False]['within_expected_range'])
        all_cyclical_valid = all(results_df[results_df['column'].str.contains('_sin|_cos')]['within_expected_range'])
        
        return {
            'all_normalized': all_normalized,
            'all_cyclical_valid': all_cyclical_valid,
            'stats': results_df.to_dict('records'),
            'results_file': results_file,
            'plot_file': plot_file
        }
    
    def test_data_recovery(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame, 
                          scaling_params_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Test if original data can be recovered from normalized data.
        
        Args:
            raw_df: Original data
            processed_df: Normalized data
            scaling_params_file: Optional file with scaling parameters
            
        Returns:
            Dictionary with test results
        """
        logging.info("Testing data recovery from normalized data")
        
        recovery_results = {
            'column': [],
            'recovery_error': [],
            'recovery_error_pct': [],
            'max_error': [],
            'recoverable': []
        }
        
        # Load scaling parameters if available
        scaling_params = None
        if scaling_params_file and os.path.exists(scaling_params_file):
            with open(scaling_params_file, 'r') as f:
                scaling_params = json.load(f)
            logging.info(f"Loaded scaling parameters from {scaling_params_file}")
        
        # For each numeric column in processed data, attempt to recover original values
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        cyclical_cols = [col for col in self.expected_cyclical_cols if col in processed_df.columns]
        cols_to_test = [col for col in numeric_cols 
                        if col not in cyclical_cols 
                        and col in raw_df.columns 
                        and col != 'inTheMoney']
        
        for col in cols_to_test:
            logging.info(f"Testing recovery for column: {col}")
            
            # Get normalized values
            norm_values = processed_df[col].dropna()
            
            # Get original values
            orig_values = raw_df.loc[processed_df.index, col].dropna()
            
            # Only test where both have values
            common_idx = norm_values.index.intersection(orig_values.index)
            if len(common_idx) == 0:
                logging.warning(f"No common indices for column {col}, skipping")
                continue
                
            norm_values = norm_values.loc[common_idx]
            orig_values = orig_values.loc[common_idx]
            
            # Attempt to recover original values
            if scaling_params and col in scaling_params:
                # Use stored parameters
                mean = scaling_params[col]['mean']
                std = scaling_params[col]['std']
                logging.info(f"Using scaling parameters: mean={mean}, std={std}")
                
                # Reconstruct original values
                recovered_values = norm_values * std + mean
            else:
                # Estimate parameters from original data
                mean = orig_values.mean()
                std = orig_values.std()
                logging.info(f"Estimated scaling parameters: mean={mean}, std={std}")
                
                # Reconstruct original values
                recovered_values = norm_values * std + mean
            
            # Calculate recovery error
            abs_error = np.abs(recovered_values - orig_values)
            mean_abs_error = abs_error.mean()
            max_abs_error = abs_error.max()
            
            # Calculate percentage error
            mean_abs_pct_error = (abs_error / (np.abs(orig_values) + 1e-10)).mean() * 100
            
            # Determine if recovery is acceptable (< 1% error)
            is_recoverable = mean_abs_pct_error < 1.0
            
            # Add to results
            recovery_results['column'].append(col)
            recovery_results['recovery_error'].append(mean_abs_error)
            recovery_results['recovery_error_pct'].append(mean_abs_pct_error)
            recovery_results['max_error'].append(max_abs_error)
            recovery_results['recoverable'].append(is_recoverable)
            
            logging.info(f"Column {col}: mean_abs_error={mean_abs_error:.6f}, "
                        f"mean_pct_error={mean_abs_pct_error:.2f}%, "
                        f"max_error={max_abs_error:.6f}, "
                        f"recoverable={is_recoverable}")
            
            # Plot comparison for a few points (first 100 values)
            if len(orig_values) > 10:
                plt.figure(figsize=(12, 6))
                
                sample_size = min(100, len(orig_values))
                idx = orig_values.index[:sample_size]
                
                plt.plot(orig_values.loc[idx], label='Original', linewidth=2)
                plt.plot(recovered_values.loc[idx], label='Recovered', linewidth=1, linestyle='--')
                
                plt.title(f'Data Recovery Test for {col}')
                plt.xlabel('Data Point Index')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_file = os.path.join(self.output_dir, f'recovery_plot_{col}.png')
                plt.savefig(plot_file)
                plt.close()
        
        # Create results DataFrame
        results_df = pd.DataFrame(recovery_results)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'recovery_test.csv')
        results_df.to_csv(results_file, index=False)
        logging.info(f"Saved recovery test results to {results_file}")
        
        # Create summary plot
        plt.figure(figsize=(12, 6))
        plt.bar(results_df['column'], results_df['recovery_error_pct'], alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.title('Data Recovery Test: Percentage Error by Column')
        plt.ylabel('Mean Absolute Percentage Error (%)')
        plt.axhline(y=1.0, color='r', linestyle='--', label='1% Error Threshold')
        plt.legend()
        plt.tight_layout()
        
        summary_plot = os.path.join(self.output_dir, 'recovery_summary.png')
        plt.savefig(summary_plot)
        plt.close()
        
        # Check overall result
        all_recoverable = all(results_df['recoverable'])
        
        return {
            'all_recoverable': all_recoverable,
            'stats': results_df.to_dict('records'),
            'results_file': results_file,
            'summary_plot': summary_plot
        }
    
    def test_cyclical_features(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test if cyclical features are correctly encoded.
        
        Args:
            processed_df: DataFrame with processed data
            
        Returns:
            Dictionary with test results
        """
        logging.info("Testing cyclical feature encoding")
        
        # List of expected cyclical feature pairs
        cyclical_pairs = [
            ('day_of_week_sin', 'day_of_week_cos', 'day_of_week', 7),
            ('day_of_month_sin', 'day_of_month_cos', 'day_of_month', 31),
            ('day_of_year_sin', 'day_of_year_cos', 'day_of_year', 365)
        ]
        
        results = {
            'feature_pair': [],
            'sin_min': [],
            'sin_max': [],
            'cos_min': [],
            'cos_max': [],
            'cycle_complete': [],
            'unit_circle_valid': []
        }
        
        for sin_col, cos_col, orig_col, period in cyclical_pairs:
            # Check if columns exist
            if sin_col not in processed_df.columns or cos_col not in processed_df.columns:
                logging.warning(f"Cyclical pair {sin_col}/{cos_col} not found in data")
                continue
                
            # Get values and drop NaNs
            sin_vals = processed_df[sin_col].dropna()
            cos_vals = processed_df[cos_col].dropna()
            
            # Verify range
            sin_min, sin_max = sin_vals.min(), sin_vals.max()
            cos_min, cos_max = cos_vals.min(), cos_vals.max()
            
            # Check if values form a unit circle
            # Calculate sin²(x) + cos²(x) which should be ≈ 1
            if len(sin_vals) == len(cos_vals):
                unit_circle_values = sin_vals**2 + cos_vals**2
                unit_circle_mean = unit_circle_values.mean()
                unit_circle_valid = 0.99 <= unit_circle_mean <= 1.01
            else:
                unit_circle_valid = False
                unit_circle_mean = None
            
            # Check if full cycle is represented
            cycle_complete = (
                -1.0 <= sin_min <= -0.9 and 
                0.9 <= sin_max <= 1.0 and
                -1.0 <= cos_min <= -0.9 and
                0.9 <= cos_max <= 1.0
            )
            
            # Store results
            results['feature_pair'].append(f"{sin_col}/{cos_col}")
            results['sin_min'].append(sin_min)
            results['sin_max'].append(sin_max)
            results['cos_min'].append(cos_min)
            results['cos_max'].append(cos_max)
            results['cycle_complete'].append(cycle_complete)
            results['unit_circle_valid'].append(unit_circle_valid)
            
            logging.info(f"Cyclical pair {sin_col}/{cos_col}:")
            logging.info(f"  Sin range: [{sin_min:.4f}, {sin_max:.4f}]")
            logging.info(f"  Cos range: [{cos_min:.4f}, {cos_max:.4f}]")
            logging.info(f"  Cycle complete: {cycle_complete}")
            logging.info(f"  Unit circle valid: {unit_circle_valid} (mean: {unit_circle_mean})")
            
            # Plot cyclical encoding if original column is available
            if orig_col in processed_df.columns:
                plt.figure(figsize=(12, 6))
                
                # Get original values
                orig_vals = processed_df[orig_col].dropna()
                
                # Get common indices
                common_idx = orig_vals.index.intersection(sin_vals.index).intersection(cos_vals.index)
                
                if len(common_idx) > 0:
                    # Use only common indices
                    orig_sample = orig_vals.loc[common_idx].iloc[:100]  # Sample first 100
                    sin_sample = sin_vals.loc[common_idx].iloc[:100]
                    cos_sample = cos_vals.loc[common_idx].iloc[:100]
                    
                    # Plot
                    plt.plot(orig_sample.index, orig_sample, 'o-', label=f'Original {orig_col}')
                    plt.plot(sin_sample.index, sin_sample, label=f'{sin_col}')
                    plt.plot(cos_sample.index, cos_sample, label=f'{cos_col}')
                    plt.title(f'Cyclical Encoding: {orig_col}')
                    plt.xlabel('Data Point Index')
                    plt.ylabel('Value')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    
                    # Save plot
                    cyclical_plot = os.path.join(self.output_dir, f'cyclical_plot_{orig_col}.png')
                    plt.savefig(cyclical_plot)
                    plt.close()
                    
                    # Plot phase space (sin vs cos)
                    plt.figure(figsize=(8, 8))
                    plt.scatter(sin_vals, cos_vals, alpha=0.1)
                    plt.title(f'Phase Space: {sin_col} vs {cos_col}')
                    plt.xlabel(sin_col)
                    plt.ylabel(cos_col)
                    plt.grid(True, alpha=0.3)
                    
                    # Add unit circle
                    theta = np.linspace(0, 2*np.pi, 100)
                    plt.plot(np.sin(theta), np.cos(theta), 'r--')
                    
                    plt.xlim(-1.1, 1.1)
                    plt.ylim(-1.1, 1.1)
                    plt.axis('equal')
                    
                    phase_plot = os.path.join(self.output_dir, f'phase_plot_{orig_col}.png')
                    plt.savefig(phase_plot)
                    plt.close()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'cyclical_test.csv')
        results_df.to_csv(results_file, index=False)
        logging.info(f"Saved cyclical test results to {results_file}")
        
        # Check overall result
        all_valid = all(results_df['unit_circle_valid']) and all(results_df['cycle_complete'])
        
        return {
            'all_valid': all_valid,
            'stats': results_df.to_dict('records'),
            'results_file': results_file
        }
    
    def test_ticker_consistency(self, processed_dir: str) -> Dict[str, Any]:
        """
        Test consistency of preprocessing across different tickers.
        
        Args:
            processed_dir: Directory containing ticker-specific files
            
        Returns:
            Dictionary with test results
        """
        logging.info(f"Testing ticker consistency in {processed_dir}")
        
        # Read ticker metadata
        metadata_file = os.path.join(processed_dir, 'ticker_metadata.csv')
        if not os.path.exists(metadata_file):
            logging.warning(f"Ticker metadata file not found: {metadata_file}")
            return {'success': False, 'error': 'Metadata file not found'}
            
        metadata = pd.read_csv(metadata_file)
        logging.info(f"Found {len(metadata)} tickers in metadata")
        
        # Test structure and columns for each ticker
        ticker_results = {
            'ticker': [],
            'file_exists': [],
            'row_count': [],
            'all_expected_columns': [],
            'column_count': [],
            'has_normalized_features': []
        }
        
        for idx, row in metadata.iterrows():
            ticker = row['ticker']
            file_path = row['file_path']
            
            # Check if file exists
            file_exists = os.path.exists(file_path)
            ticker_results['ticker'].append(ticker)
            ticker_results['file_exists'].append(file_exists)
            
            if not file_exists:
                logging.warning(f"Ticker file not found: {file_path}")
                ticker_results['row_count'].append(0)
                ticker_results['all_expected_columns'].append(False)
                ticker_results['column_count'].append(0)
                ticker_results['has_normalized_features'].append(False)
                continue
            
            # Load ticker data
            try:
                ticker_df = pd.read_csv(file_path)
                row_count = len(ticker_df)
                column_count = len(ticker_df.columns)
                
                ticker_results['row_count'].append(row_count)
                ticker_results['column_count'].append(column_count)
                
                # Check for expected columns
                numeric_cols = self.expected_numeric_cols
                cyclical_cols = [col for col in self.expected_cyclical_cols if '_sin' in col or '_cos' in col]
                expected_cols = numeric_cols + cyclical_cols
                
                found_columns = [col for col in expected_cols if col in ticker_df.columns]
                all_expected = len(found_columns) == len(expected_cols)
                ticker_results['all_expected_columns'].append(all_expected)
                
                # Check if numeric features are normalized
                if not all_expected:
                    ticker_results['has_normalized_features'].append(False)
                    continue
                
                # Test a sample of columns for normalization
                sample_cols = [col for col in numeric_cols if col in ticker_df.columns][:5]
                is_normalized = True
                
                for col in sample_cols:
                    values = ticker_df[col].dropna()
                    if len(values) == 0:
                        continue
                        
                    mean = abs(values.mean())
                    std = values.std()
                    
                    if mean > 0.1 or std < 0.9 or std > 1.1:
                        is_normalized = False
                        break
                
                ticker_results['has_normalized_features'].append(is_normalized)
                
                # Log results
                logging.info(f"Ticker {ticker}: rows={row_count}, columns={column_count}, "
                            f"all_expected_cols={all_expected}, normalized={is_normalized}")
                
            except Exception as e:
                logging.error(f"Error processing ticker {ticker}: {str(e)}")
                ticker_results['row_count'].append(0)
                ticker_results['all_expected_columns'].append(False)
                ticker_results['column_count'].append(0)
                ticker_results['has_normalized_features'].append(False)
        
        # Create results DataFrame
        results_df = pd.DataFrame(ticker_results)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'ticker_consistency_test.csv')
        results_df.to_csv(results_file, index=False)
        logging.info(f"Saved ticker consistency test results to {results_file}")
        
        # Check overall result
        all_files_exist = all(results_df['file_exists'])
        all_expected_columns = all(results_df['all_expected_columns'])
        all_normalized = all(results_df['has_normalized_features'])
        
        return {
            'all_files_exist': all_files_exist,
            'all_expected_columns': all_expected_columns,
            'all_normalized': all_normalized,
            'stats': results_df.to_dict('records'),
            'results_file': results_file
        }
    
    def run_all_tests(self, 
                      raw_file: str, 
                      processed_file: str, 
                      processed_dir: Optional[str] = None,
                      scaling_params_file: Optional[str] = None,
                      specific_ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Run all tests on the processed data.
        
        Args:
            raw_file: Path to original data file
            processed_file: Path to processed data file
            processed_dir: Optional directory containing ticker-specific files
            scaling_params_file: Optional file with scaling parameters
            specific_ticker: Optional ticker to focus testing on
            
        Returns:
            Dictionary with all test results
        """
        start_time = time.time()
        logging.info(f"Starting comprehensive testing of processed options data")
        
        all_results = {}
        
        try:
            # Load test data
            raw_df, processed_df = self.load_test_data(raw_file, processed_file, specific_ticker)
            
            # Run normalization test
            norm_results = self.test_normalization(processed_df)
            all_results['normalization'] = norm_results
            
            # Run data recovery test
            recovery_results = self.test_data_recovery(raw_df, processed_df, scaling_params_file)
            all_results['recovery'] = recovery_results
            
            # Run cyclical features test
            cyclical_results = self.test_cyclical_features(processed_df)
            all_results['cyclical'] = cyclical_results
            
            # Run ticker consistency test if directory provided
            if processed_dir and os.path.exists(processed_dir):
                ticker_results = self.test_ticker_consistency(processed_dir)
                all_results['ticker_consistency'] = ticker_results
            
            # Overall success determination
            overall_success = (
                norm_results.get('all_normalized', False) and
                recovery_results.get('all_recoverable', False) and
                cyclical_results.get('all_valid', False)
            )
            
            if 'ticker_consistency' in all_results:
                overall_success = overall_success and all_results['ticker_consistency'].get('all_files_exist', False)
            
            all_results['overall_success'] = overall_success
            
            # Calculate execution time
            execution_time = time.time() - start_time
            all_results['execution_time'] = execution_time
            
            logging.info(f"Testing completed in {execution_time:.2f} seconds")
            logging.info(f"Overall success: {overall_success}")
            
            return all_results
            
        except Exception as e:
            logging.error(f"Error running tests: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
            return {
                'overall_success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }