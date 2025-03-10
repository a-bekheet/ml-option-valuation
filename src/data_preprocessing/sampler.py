import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path
import time
from typing import Union, Optional, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('options_sampler.log')
    ]
)

class OptionsDataSampler:
    """
    A utility to load, inspect, and sample options data from .npy or .csv files.
    """
    
    def __init__(
        self, 
        input_file: str,
        output_dir: str = 'sampled_data',
        sample_size: Union[int, float] = 1000,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the sampler.
        
        Args:
            input_file (str): Path to the input file (.npy or .csv)
            output_dir (str): Directory to save output files
            sample_size (Union[int, float]): Either number of rows or fraction of data
            random_seed (Optional[int]): Random seed for reproducibility
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Column names for .npy files (based on the provided code)
        self.column_names = [
            'contractSymbol',      # 0
            'lastTradeDate',       # 1
            'strike',              # 2
            'lastPrice',           # 3
            'bid',                 # 4
            'ask',                 # 5
            'change',              # 6
            'percentChange',       # 7
            'volume',              # 8
            'openInterest',        # 9
            'impliedVolatility',   # 10
            'inTheMoney',          # 11
            'contractSize',        # 12
            'currency',            # 13
            'quoteDate',           # 14
            'expiryDate',          # 15
            'daysToExpiry',        # 16
            'stockVolume',         # 17
            'stockClose',          # 18
            'stockAdjClose',       # 19
            'stockOpen',           # 20
            'stockHigh',           # 21
            'stockLow',            # 22
            'strikeDelta',         # 23
            'stockClose_ewm_5d',   # 24
            'stockClose_ewm_15d',  # 25
            'stockClose_ewm_45d',  # 26
            'stockClose_ewm_135d'  # 27
        ]
    
    def load_file(self) -> pd.DataFrame:
        """
        Load data from the input file and return as a DataFrame.
        
        Returns:
            pd.DataFrame: The loaded data
        """
        start_time = time.time()
        logging.info(f"Loading data from {self.input_file}")
        
        try:
            # Check input file extension
            file_ext = os.path.splitext(self.input_file)[1].lower()
            
            if file_ext == '.npy':
                # Load .npy file
                data = np.load(self.input_file, allow_pickle=True)
                logging.info(f"Loaded NumPy array of shape {data.shape}")
                
                # Check if the number of columns matches our expected columns
                if data.shape[1] == len(self.column_names):
                    df = pd.DataFrame(data, columns=self.column_names)
                    logging.info(f"Created DataFrame with columns: {self.column_names}")
                else:
                    logging.warning(
                        f"Column count mismatch: array has {data.shape[1]} columns, "
                        f"expected {len(self.column_names)}"
                    )
                    # Create DataFrame with as many columns as possible
                    if data.shape[1] < len(self.column_names):
                        columns = self.column_names[:data.shape[1]]
                    else:
                        columns = self.column_names + [f"Unknown{i}" for i in range(data.shape[1] - len(self.column_names))]
                    
                    df = pd.DataFrame(data, columns=columns)
                    logging.info(f"Created DataFrame with adjusted columns: {list(df.columns)}")
            
            elif file_ext == '.csv':
                # Load CSV file
                df = pd.read_csv(self.input_file)
                logging.info(f"Loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
            
            else:
                logging.error(f"Unsupported file type: {file_ext}")
                raise ValueError(f"Unsupported file type: {file_ext}. Please provide a .npy or .csv file.")
                
            # Record loading time
            load_time = time.time() - start_time
            logging.info(f"Data loaded in {load_time:.2f} seconds")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading file: {str(e)}")
            raise
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the loaded data to get basic information.
        
        Args:
            df (pd.DataFrame): The data to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logging.info("Analyzing data...")
        
        try:
            # Get basic statistics
            total_rows = len(df)
            total_cols = len(df.columns)
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            missing_values = df.isna().sum().sum()
            
            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            
            # Calculate column types
            col_types = df.dtypes.value_counts().to_dict()
            col_type_str = ', '.join([f"{k}: {v}" for k, v in col_types.items()])
            
            # Check for missing columns
            expected_cols = set(self.column_names)
            actual_cols = set(df.columns)
            missing_cols = list(expected_cols - actual_cols)
            extra_cols = list(actual_cols - expected_cols)
            
            # Check for the 'ticker' column
            has_ticker = 'ticker' in df.columns
            
            # If no ticker column, extract it from contractSymbol
            if not has_ticker and 'contractSymbol' in df.columns:
                # Try to extract ticker from the first few rows
                sample_symbols = df['contractSymbol'].head(5).tolist()
                logging.info(f"Sample contract symbols: {sample_symbols}")
                
                # Extract ticker using regex if possible
                try:
                    import re
                    # Extract ticker letters at the beginning
                    sample_tickers = [re.match(r'^([A-Z]+)', symbol).group(1) 
                                      for symbol in sample_symbols if pd.notna(symbol)]
                    if sample_tickers:
                        logging.info(f"Sample extracted tickers: {sample_tickers}")
                except:
                    logging.warning("Could not extract tickers from contractSymbol")
            
            # Get value counts for contractSymbol to check tickers
            if 'contractSymbol' in df.columns:
                # Get unique prefixes (potential tickers)
                import re
                symbol_prefixes = df['contractSymbol'].dropna().apply(
                    lambda x: re.match(r'^([A-Z]+)', str(x)).group(1) if re.match(r'^([A-Z]+)', str(x)) else None
                ).dropna().value_counts()
                
                top_prefixes = symbol_prefixes.head(10).to_dict()
                logging.info(f"Top symbol prefixes (likely tickers): {top_prefixes}")
            
            # Summary of analysis
            analysis = {
                'total_rows': total_rows,
                'total_columns': total_cols,
                'memory_usage_mb': memory_usage,
                'missing_values': missing_values,
                'duplicate_rows': duplicates,
                'column_types': col_types,
                'missing_expected_columns': missing_cols,
                'extra_columns': extra_cols,
                'has_ticker_column': has_ticker
            }
            
            # Log analysis results
            logging.info(f"Data shape: {total_rows} rows x {total_cols} columns")
            logging.info(f"Memory usage: {memory_usage:.2f} MB")
            logging.info(f"Missing values: {missing_values}")
            logging.info(f"Duplicate rows: {duplicates}")
            logging.info(f"Column types: {col_type_str}")
            
            if missing_cols:
                logging.warning(f"Missing expected columns: {missing_cols}")
            if extra_cols:
                logging.info(f"Extra columns found: {extra_cols}")
                
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing data: {str(e)}")
            raise
    
    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a sample of the data.
        
        Args:
            df (pd.DataFrame): The full dataset
            
        Returns:
            pd.DataFrame: The sampled data
        """
        logging.info("Creating data sample...")
        
        try:
            total_rows = len(df)
            
            # Set random seed for reproducibility
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            
            # Determine sample size
            if isinstance(self.sample_size, float):
                if 0 < self.sample_size < 1:
                    sample_rows = int(total_rows * self.sample_size)
                else:
                    sample_rows = int(self.sample_size)
            else:
                sample_rows = min(self.sample_size, total_rows)
                
            logging.info(f"Sampling {sample_rows} rows from {total_rows} total rows")
            
            # Create the sample
            if 'ticker' in df.columns:
                # Stratified sampling by ticker
                logging.info("Performing stratified sampling by ticker")
                
                # Calculate proportions for each ticker
                ticker_counts = df['ticker'].value_counts(normalize=True)
                
                sampled_dfs = []
                for ticker, proportion in ticker_counts.items():
                    # Calculate sample size for this ticker
                    ticker_sample_size = max(1, int(sample_rows * proportion))
                    ticker_df = df[df['ticker'] == ticker]
                    
                    # Sample min of calculated size or available rows
                    actual_sample_size = min(ticker_sample_size, len(ticker_df))
                    ticker_sample = ticker_df.sample(n=actual_sample_size)
                    sampled_dfs.append(ticker_sample)
                    
                    logging.info(f"Sampled {actual_sample_size} rows for ticker {ticker}")
                
                # Combine all sampled tickers
                df_sample = pd.concat(sampled_dfs, ignore_index=True)
                
                # Shuffle the final sample
                df_sample = df_sample.sample(frac=1).reset_index(drop=True)
                
            else:
                # Simple random sampling
                logging.info("Performing simple random sampling")
                df_sample = df.sample(n=sample_rows).reset_index(drop=True)
            
            logging.info(f"Final sample shape: {df_sample.shape}")
            return df_sample
            
        except Exception as e:
            logging.error(f"Error sampling data: {str(e)}")
            raise
    
    def extract_ticker_from_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ticker information from contractSymbol if ticker column doesn't exist.
        
        Args:
            df (pd.DataFrame): The DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with ticker column added
        """
        if 'ticker' not in df.columns and 'contractSymbol' in df.columns:
            logging.info("Extracting ticker from contractSymbol")
            
            try:
                import re
                df['ticker'] = df['contractSymbol'].astype(str).apply(
                    lambda x: re.match(r'^([A-Z]+)', x).group(1) if re.match(r'^([A-Z]+)', x) else None
                )
                
                # Check the results
                ticker_counts = df['ticker'].value_counts().head(10).to_dict()
                logging.info(f"Extracted tickers (top 10): {ticker_counts}")
                
                # Check for None values
                none_count = df['ticker'].isna().sum()
                if none_count > 0:
                    logging.warning(f"Could not extract ticker for {none_count} rows")
                
                return df
                
            except Exception as e:
                logging.error(f"Error extracting tickers: {str(e)}")
                # Return original DataFrame if extraction fails
                return df
        else:
            return df
    
    def save_sample(self, df: pd.DataFrame) -> str:
        """
        Save the sampled data to a CSV file.
        
        Args:
            df (pd.DataFrame): The data to save
            
        Returns:
            str: Path to the saved file
        """
        filename = os.path.basename(self.input_file)
        base_name = os.path.splitext(filename)[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        output_file = os.path.join(self.output_dir, f"{base_name}_sample_{len(df)}rows_{timestamp}.csv")
        
        logging.info(f"Saving sample to {output_file}")
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def export_column_info(self, df: pd.DataFrame) -> str:
        """
        Export detailed information about each column.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze
            
        Returns:
            str: Path to the saved information file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        info_file = os.path.join(self.output_dir, f"column_info_{timestamp}.csv")
        
        # Create column information
        column_info = []
        
        for col in df.columns:
            # Basic information
            col_data = df[col]
            data_type = str(col_data.dtype)
            missing = col_data.isna().sum()
            present = len(df) - missing
            pct_present = (present / len(df)) * 100 if len(df) > 0 else 0
            
            # Calculate unique values
            unique_count = col_data.nunique()
            pct_unique = (unique_count / present) * 100 if present > 0 else 0
            
            # Get sample values (non-null)
            sample_values = col_data.dropna().head(3).tolist()
            sample_str = str(sample_values)[:100]  # Truncate long samples
            
            # Type-specific information
            if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_categorical_dtype(col_data):
                min_val = col_data.min() if not col_data.isna().all() else None
                max_val = col_data.max() if not col_data.isna().all() else None
                mean_val = col_data.mean() if not col_data.isna().all() else None
                std_val = col_data.std() if not col_data.isna().all() else None
                
                stats = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val
                }
            else:
                stats = None
            
            # Add to list
            column_info.append({
                'column_name': col,
                'data_type': data_type,
                'present_count': present,
                'missing_count': missing,
                'percent_present': round(pct_present, 2),
                'unique_values': unique_count,
                'percent_unique': round(pct_unique, 2),
                'sample_values': sample_str,
                'statistics': str(stats) if stats else None
            })
        
        # Convert to DataFrame and save
        column_df = pd.DataFrame(column_info)
        column_df.to_csv(info_file, index=False)
        
        logging.info(f"Saved column information to {info_file}")
        return info_file
    
    def process(self) -> Dict[str, Any]:
        """
        Execute the complete sampling process.
        
        Returns:
            Dict[str, Any]: Results of the processing
        """
        try:
            # Step 1: Load the data
            df = self.load_file()
            
            # Step 2: Analyze the data
            analysis = self.analyze_data(df)
            
            # Step 3: Extract ticker if needed
            df = self.extract_ticker_from_symbol(df)
            
            # Step 4: Sample the data
            sample_df = self.sample_data(df)
            
            # Step 5: Export column information
            info_file = self.export_column_info(sample_df)
            
            # Step 6: Save the sample
            sample_file = self.save_sample(sample_df)
            
            # Return processing results
            return {
                'original_shape': df.shape,
                'sample_shape': sample_df.shape,
                'analysis': analysis,
                'sample_file': sample_file,
                'info_file': info_file
            }
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'error': str(e),
                'traceback': traceback.format_exc()
            }

def main():
    """
    Main function to run the options data sampler.
    """
    print("Options Data Sampler")
    print("=" * 50)
    
    # Get input file path
    input_file = input("Enter path to options data file (.npy or .csv): ")
    if not input_file:
        input_file = "data_files_new/option_data_with_headers.npy"
        print(f"Using default path: {input_file}")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return
    
    # Get sample size
    try:
        sample_size = input("Enter sample size (default: 1000): ")
        if not sample_size:
            sample_size = 1000
        else:
            # Try to convert to int or float
            if '.' in sample_size:
                sample_size = float(sample_size)
            else:
                sample_size = int(sample_size)
    except ValueError:
        sample_size = 1000
        print(f"Invalid input. Using default sample size: {sample_size}")
    
    # Create sampler
    sampler = OptionsDataSampler(
        input_file=input_file,
        output_dir="sampled_data",
        sample_size=sample_size,
        random_seed=42
    )
    
    # Run sampling process
    print("\nProcessing data...")
    results = sampler.process()
    
    # Display results
    if 'error' in results:
        print(f"\nError: {results['error']}")
        print("See options_sampler.log for detailed error information.")
    else:
        print("\nProcessing completed successfully!")
        print("=" * 50)
        print(f"Original data shape: {results['original_shape']}")
        print(f"Sample data shape: {results['sample_shape']}")
        print(f"\nSample saved to: {results['sample_file']}")
        print(f"Column information saved to: {results['info_file']}")
        print("\nSee options_sampler.log for detailed processing information.")

if __name__ == "__main__":
    main()