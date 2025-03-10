import pandas as pd
import numpy as np
from typing import Union, Optional
import logging
from datetime import datetime
import os

class OptionsDataSampler:
    def __init__(
        self, 
        input_file: str,
        output_dir: str = 'data_insights/csv_files',
        sample_size: Union[int, float] = 0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the sampler.
        
        Args:
            input_file (str): Path to input CSV file
            output_dir (str): Directory to save output files
            sample_size (Union[int, float]): Either number of rows or fraction of data
            random_seed (Optional[int]): Random seed for reproducibility
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_sample(self, stratify_by: Optional[str] = 'ticker') -> pd.DataFrame:
        """
        Extract a random sample from the options data file.
        
        Args:
            stratify_by (Optional[str]): Column to use for stratified sampling
                                       'ticker' - Maintain ticker proportions
                                       None - Simple random sampling
        """
        try:
            # Read the data
            self.logger.info(f"Reading data from {self.input_file}")
            df = pd.read_csv(self.input_file)
            
            total_rows = len(df)
            self.logger.info(f"Total rows in dataset: {total_rows:,}")
            
            # Extract ticker from contractSymbol if stratifying
            if stratify_by == 'ticker':
                df['ticker'] = df['contractSymbol'].str.extract(r'([A-Z]+)')
            
            # Calculate sample size if fraction
            if isinstance(self.sample_size, float):
                n_samples = int(total_rows * self.sample_size)
            else:
                n_samples = min(self.sample_size, total_rows)
            
            # Set random seed for reproducibility
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            
            # Perform stratified sampling if requested
            if stratify_by:
                self.logger.info(f"Performing stratified sampling by {stratify_by}")
                
                # Calculate proportions for each stratum
                proportions = df[stratify_by].value_counts(normalize=True)
                
                # Sample from each stratum
                sampled_dfs = []
                for stratum, prop in proportions.items():
                    stratum_size = int(np.ceil(n_samples * prop))
                    stratum_df = df[df[stratify_by] == stratum]
                    
                    # Sample min of calculated size or available rows
                    sample_size = min(stratum_size, len(stratum_df))
                    stratum_sample = stratum_df.sample(n=sample_size)
                    sampled_dfs.append(stratum_sample)
                
                sample_df = pd.concat(sampled_dfs, ignore_index=True)
                
                # Shuffle the final sample
                sample_df = sample_df.sample(frac=1).reset_index(drop=True)
                
            else:
                self.logger.info("Performing simple random sampling")
                sample_df = df.sample(n=n_samples)
            
            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Save the sample
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.output_dir,
                f'options_sample_{n_samples}rows_{timestamp}.csv'
            )
            
            # Drop the temporary ticker column if it was added
            if stratify_by == 'ticker' and 'ticker' not in df.columns:
                sample_df = sample_df.drop(columns=['ticker'])
            
            sample_df.to_csv(output_file, index=False)
            
            # Log sampling results
            self.logger.info(f"\nSampling Summary:")
            self.logger.info(f"Original dataset size: {total_rows:,} rows")
            self.logger.info(f"Sample size: {len(sample_df):,} rows")
            if stratify_by:
                original_tickers = df['contractSymbol'].str.extract(r'([A-Z]+)')
                sampled_tickers = sample_df['contractSymbol'].str.extract(r'([A-Z]+)')
                
                self.logger.info(f"\nTicker proportions comparison:")
                orig_props = original_tickers[0].value_counts(normalize=True)
                sample_props = sampled_tickers[0].value_counts(normalize=True)
                
                for ticker in orig_props.index:
                    orig_pct = orig_props.get(ticker, 0) * 100
                    sample_pct = sample_props.get(ticker, 0) * 100
                    self.logger.info(
                        f"{ticker}: {sample_pct:.1f}% in sample "
                        f"(vs {orig_pct:.1f}% in original)"
                    )
            
            self.logger.info(f"\nSample saved to: {output_file}")
            
            return sample_df
            
        except Exception as e:
            self.logger.error(f"Error during sampling: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize sampler
    sampler = OptionsDataSampler(
        input_file='data_files/option_data.csv',
        sample_size=100,  # 10% sample
        random_seed=42    # for reproducibility
    )
    
    # Extract sample
    # For simple random sampling:
    sample_df = sampler.extract_sample(stratify_by=None)
    
    # For stratified sampling by ticker:
    # sample_df = sampler.extract_sample(stratify_by='ticker')