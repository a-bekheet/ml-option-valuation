#!/usr/bin/env python3
"""
Comprehensive Options Data Preprocessing Script
- Drops incomplete features and invalid rows
- Optimizes data types for memory efficiency
- Handles stock splits
- Feature engineers option greeks
- Performs time series encoding
- Splits data by ticker for model training
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OptionsPreprocessor:
    def __init__(self, input_file, rates_file="reprocessing/DGS10.csv", output_dir="processed_data", min_completion_pct=50):
        """
        Initialize the Options Data Preprocessor
        
        Args:
            input_file: Path to the raw data CSV file
            rates_file: Path to the risk-free rates CSV file (DGS10)
            output_dir: Directory to save processed files
            min_completion_pct: Minimum percentage of non-null values required to keep a column
        """
        self.input_file = input_file
        self.rates_file = rates_file
        self.output_dir = output_dir
        self.min_completion_pct = min_completion_pct
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Stock split information
        self.splits = {
            'TSLA': {
                'date': '2022-08-25',
                'ratio': 3.0,
                'type': 'forward',
                'adjustments': {
                    'divide': [  # Fields to divide by ratio (pre-split values)
                        'strike', 'lastPrice', 'bid', 'ask',
                        'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                        'stockClose_ewm_5d', 'stockClose_ewm_15d',
                        'stockClose_ewm_45d', 'stockClose_ewm_135d',
                        'strikeDelta'
                    ],
                    'multiply': [  # Fields to multiply by ratio (pre-split values)
                        'volume', 'openInterest', 'stockVolume'
                    ]
                }
            },
            'SHOP': {
                'date': '2022-07-04',
                'ratio': 10.0,
                'type': 'forward',
                'adjustments': {
                    'divide': [
                        'strike', 'lastPrice', 'bid', 'ask',
                        'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                        'stockClose_ewm_5d', 'stockClose_ewm_15d',
                        'stockClose_ewm_45d', 'stockClose_ewm_135d',
                        'strikeDelta'
                    ],
                    'multiply': [
                        'volume', 'openInterest', 'stockVolume'
                    ]
                }
            },
            'GOOGL': {
                'date': '2022-07-18',
                'ratio': 20.0,
                'type': 'forward',
                'adjustments': {
                    'divide': [
                        'strike', 'lastPrice', 'bid', 'ask',
                        'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                        'stockClose_ewm_5d', 'stockClose_ewm_15d',
                        'stockClose_ewm_45d', 'stockClose_ewm_135d',
                        'strikeDelta'
                    ],
                    'multiply': [
                        'volume', 'openInterest', 'stockVolume'
                    ]
                }
            },
            'CGC': {
                'date': '2023-12-20',
                'ratio': 0.1,  # 1:10 reverse split
                'type': 'reverse',
                'adjustments': {
                    'multiply': [  # For reverse split, we multiply price fields
                        'strike', 'lastPrice', 'bid', 'ask',
                        'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                        'stockClose_ewm_5d', 'stockClose_ewm_15d',
                        'stockClose_ewm_45d', 'stockClose_ewm_135d',
                        'strikeDelta'
                    ],
                    'divide': [  # For reverse split, we divide volume fields
                        'volume', 'openInterest', 'stockVolume'
                    ]
                }
            }
        }
        
        # Fields to optimize by datatype
        self.dtype_optimizations = {
            'float32': [
                'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange',
                'impliedVolatility', 'stockClose', 'stockHigh', 'stockLow', 'stockOpen',
                'stockAdjClose', 'strikeDelta', 'stockClose_ewm_5d', 'stockClose_ewm_15d',
                'stockClose_ewm_45d', 'stockClose_ewm_135d'
            ],
            'uint32': ['volume', 'openInterest', 'stockVolume'],
            'category': ['contractSize', 'currency', 'ticker'],
            'datetime': ['lastTradeDate', 'quoteDate', 'expiryDate']
        }
        
        # Required columns for the model (to be kept)
        self.required_columns = [
            'contractSymbol', 'lastTradeDate', 'strike', 'bid', 'ask',
            'impliedVolatility', 'expiryDate', 'daysToExpiry', 'stockClose',
            'stockVolume', 'ticker'
        ]
        
        # Initialize metrics
        self.metrics = {
            'rows_initial': 0,
            'rows_after_cleaning': 0,
            'rows_after_filtering': 0,
            'columns_initial': 0,
            'columns_dropped': 0,
            'memory_initial': 0,
            'memory_final': 0
        }
    
    def load_data(self):
        """Load data from either CSV or NPY file and record initial metrics"""
        print(f"Loading data from {self.input_file}...")
        start_time = time.time()
        
        # Check file extension
        file_extension = os.path.splitext(self.input_file)[1].lower()
        
        if file_extension == '.npy':
            # Load from NumPy file
            raw_data = np.load(self.input_file, allow_pickle=True)
            
            # Define column names based on the expected structure
            # You'll need to adjust these column names to match your data structure
            column_names = [
                'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
                'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility',
                'inTheMoney', 'contractSize', 'currency', 'quoteDate', 'expiryDate',
                'daysToExpiry', 'stockVolume', 'stockClose', 'stockAdjClose', 'stockOpen',
                'stockHigh', 'stockLow', 'strikeDelta', 'stockClose_ewm_5d',
                'stockClose_ewm_15d', 'stockClose_ewm_45d', 'stockClose_ewm_135d', 'ticker'
            ]
            
            # Convert NumPy array to DataFrame
            self.data = pd.DataFrame(raw_data, columns=column_names)
            print(f"Loaded NumPy array with shape {raw_data.shape}")
        
        elif file_extension == '.csv':
            # Load from CSV
            self.data = pd.read_csv(self.input_file)
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only .csv and .npy files are supported.")
        
        self.metrics['rows_initial'] = len(self.data)
        self.metrics['columns_initial'] = len(self.data.columns)
        self.metrics['memory_initial'] = self.data.memory_usage(deep=True).sum() / (1024**2)
        
        print(f"Loaded {self.metrics['rows_initial']:,} rows with {self.metrics['columns_initial']} columns")
        print(f"Initial memory usage: {self.metrics['memory_initial']:.2f} MB")
        print(f"Loading took {time.time() - start_time:.2f} seconds")
        
        return self.data
    
    def analyze_data_quality(self):
        """Analyze data quality and completeness"""
        print("\nAnalyzing data quality...")
        
        # Calculate completeness for each column
        completeness = pd.DataFrame({
            'column': self.data.columns,
            'dtype': self.data.dtypes,
            'non_null': self.data.notna().sum(),
            'null_count': self.data.isna().sum(),
        })
        
        completeness['completion_pct'] = (completeness['non_null'] / len(self.data)) * 100
        completeness['unique_count'] = [self.data[col].nunique() for col in self.data.columns]
        
        # Identify columns to drop
        drop_cols = []
        
        # Columns with too many nulls
        low_completion = completeness[completeness['completion_pct'] < self.min_completion_pct]
        if not low_completion.empty:
            print("\nColumns with less than 50% completion (will be dropped):")
            print(low_completion[['column', 'completion_pct']].to_string(index=False))
            drop_cols.extend(low_completion['column'].tolist())
        
        # Columns with only one unique value
        single_value = completeness[completeness['unique_count'] == 1]
        if not single_value.empty:
            print("\nColumns with only one unique value (will be dropped):")
            print(single_value[['column', 'unique_count']].to_string(index=False))
            drop_cols.extend(single_value['column'].tolist())
        
        # Keep required columns even if they would be dropped
        drop_cols = [col for col in drop_cols if col not in self.required_columns]
        
        self.columns_to_drop = drop_cols
        self.completeness = completeness
        
        print(f"\nIdentified {len(drop_cols)} columns to drop")
        return completeness
    
    def drop_incomplete_columns(self):
        """Drop columns with low completion or only one unique value"""
        if not hasattr(self, 'columns_to_drop'):
            self.analyze_data_quality()
            
        print(f"\nDropping {len(self.columns_to_drop)} columns...")
        
        if self.columns_to_drop:
            self.data = self.data.drop(columns=self.columns_to_drop)
            
        self.metrics['columns_dropped'] = len(self.columns_to_drop)
        print(f"Remaining columns: {len(self.data.columns)}")
        
        return self.data
    
    def clean_data(self):
        """Clean the data and drop rows with missing values in critical columns"""
        print("\nCleaning data...")
        
        # Handle 'daysToExpiry' - some sources have it as a string like '4 days'
        if 'daysToExpiry' in self.data.columns:
            if self.data['daysToExpiry'].dtype == 'object':
                self.data['daysToExpiry'] = self.data['daysToExpiry'].str.replace(' days', '').astype('float')
        
        # Convert boolean strings to actual booleans if needed
        if 'inTheMoney' in self.data.columns and self.data['inTheMoney'].dtype == 'object':
            self.data['inTheMoney'] = self.data['inTheMoney'].map({'True': True, 'False': False})
        
        # Drop rows with missing values in critical columns
        critical_columns = ['strike', 'bid', 'ask', 'ticker', 'stockClose', 'expiryDate']
        critical_columns = [col for col in critical_columns if col in self.data.columns]
        
        rows_before = len(self.data)
        self.data = self.data.dropna(subset=critical_columns)
        rows_after = len(self.data)
        
        print(f"Dropped {rows_before - rows_after:,} rows with missing critical values")
        self.metrics['rows_after_cleaning'] = rows_after
        
        return self.data
    
    def filter_data(self):
        """Filter data based on business rules"""
        print("\nFiltering data based on business rules...")
        
        rows_before = len(self.data)
        
        # Filter out options with unreasonable values
        if all(col in self.data.columns for col in ['bid', 'ask']):
            # Filter out options with zero or negative bid/ask
            self.data = self.data[(self.data['bid'] > 0) & (self.data['ask'] > 0)]
            
            # Filter out options with unreasonable bid-ask spread
            self.data['spread'] = self.data['ask'] - self.data['bid']
            self.data['spread_pct'] = self.data['spread'] / self.data['bid']
            self.data = self.data[self.data['spread_pct'] < 2.0]  # Remove if spread > 200% of bid
        
        # Filter out options with unreasonable implied volatility (if present)
        if 'impliedVolatility' in self.data.columns:
            self.data = self.data[
                (self.data['impliedVolatility'] > 0) & 
                (self.data['impliedVolatility'] < 5)  # Max 500% IV
            ]
        
        rows_after = len(self.data)
        print(f"Filtered out {rows_before - rows_after:,} rows with unreasonable values")
        self.metrics['rows_after_filtering'] = rows_after
        
        return self.data
    
    def apply_stock_splits(self):
        """Apply stock split adjustments"""
        print("\nApplying stock split adjustments...")
        
        # Convert quoteDate to datetime if needed
        if self.data['quoteDate'].dtype != 'datetime64[ns]':
            self.data['quoteDate'] = pd.to_datetime(self.data['quoteDate'])
        
        splits_applied = 0
        
        # Loop through each ticker with a split
        for ticker, split_info in self.splits.items():
            split_date = pd.to_datetime(split_info['date'])
            split_ratio = split_info['ratio']
            
            # Filter data for this ticker and pre-split dates
            ticker_mask = self.data['ticker'] == ticker
            pre_split_mask = ticker_mask & (self.data['quoteDate'] < split_date)
            rows_affected = pre_split_mask.sum()
            
            if rows_affected == 0:
                print(f"  No records found before {split_info['date']} for {ticker}")
                continue
            
            print(f"  Applying {split_info['type']} split ({split_ratio}) for {ticker} on {split_info['date']} - {rows_affected:,} rows affected")
            
            # Apply adjustments to fields that need division
            for field in split_info['adjustments']['divide']:
                if field in self.data.columns:
                    self.data.loc[pre_split_mask, field] = self.data.loc[pre_split_mask, field] / split_ratio
            
            # Apply adjustments to fields that need multiplication
            for field in split_info['adjustments']['multiply']:
                if field in self.data.columns:
                    self.data.loc[pre_split_mask, field] = self.data.loc[pre_split_mask, field] * split_ratio
            
            splits_applied += 1
        
        print(f"Applied {splits_applied} stock splits")
        return self.data
    
    def validate_stock_split_adjustments(self):
        """
        Validate that stock split adjustments were applied correctly
        by checking for discontinuities around split dates
        """
        print("\nValidating stock split adjustments...")
        
        issues_found = False
        
        for ticker, split_info in self.splits.items():
            # Check if this ticker exists in the data
            if ticker not in self.data['ticker'].unique():
                continue
                
            split_date = pd.to_datetime(split_info['date'])
            ticker_data = self.data[self.data['ticker'] == ticker].copy()
            
            if len(ticker_data) < 10:
                continue
                
            # Sort by date
            ticker_data = ticker_data.sort_values('quoteDate')
            
            # Get data just before and after split
            pre_split = ticker_data[ticker_data['quoteDate'] < split_date].tail(5)
            post_split = ticker_data[ticker_data['quoteDate'] >= split_date].head(5)
            
            if len(pre_split) == 0 or len(post_split) == 0:
                continue
            
            # Check for discontinuities in key metrics
            for metric in ['strike', 'stockClose']:
                if metric not in ticker_data.columns:
                    continue
                    
                pre_mean = pre_split[metric].mean()
                post_mean = post_split[metric].mean()
                
                # Skip if mean is zero
                if pre_mean == 0 or post_mean == 0:
                    continue
                    
                # Calculate ratio
                ratio = post_mean / pre_mean
                expected_ratio = 1.0  # After adjustment, we expect continuity
                
                # Allow for some variation (10%)
                if abs(ratio - expected_ratio) > 0.1:
                    print(f"  Split adjustment issue for {ticker} ({metric}): pre_mean={pre_mean:.2f}, post_mean={post_mean:.2f}, ratio={ratio:.2f}")
                    issues_found = True
        
        if not issues_found:
            print("  ✓ No split adjustment issues detected")
            
        return not issues_found
    
    def optimize_datatypes(self):
        """Optimize data types for memory efficiency"""
        print("\nOptimizing data types...")
        
        memory_before = self.data.memory_usage(deep=True).sum() / (1024**2)
        
        # 1. Convert date columns to datetime
        date_columns = self.dtype_optimizations['datetime']
        date_columns = [col for col in date_columns if col in self.data.columns]
        
        for col in date_columns:
            self.data[col] = pd.to_datetime(self.data[col])
        
        # 2. Convert float columns to float32
        float32_columns = self.dtype_optimizations['float32']
        float32_columns = [col for col in float32_columns if col in self.data.columns]
        
        for col in float32_columns:
            if self.data[col].dtype != 'float32':
                self.data[col] = self.data[col].astype('float32')
        
        # 3. Convert integer columns to unsigned integers
        uint32_columns = self.dtype_optimizations['uint32']
        uint32_columns = [col for col in uint32_columns if col in self.data.columns]
        
        for col in uint32_columns:
            # Fill NaN with 0 and convert
            self.data[col] = self.data[col].fillna(0)
            # Ensure all values are non-negative
            self.data[col] = self.data[col].clip(lower=0)
            # Convert to unsigned integer
            self.data[col] = self.data[col].astype('uint32')
        
        # 4. Convert categorical columns
        cat_columns = self.dtype_optimizations['category']
        cat_columns = [col for col in cat_columns if col in self.data.columns]
        
        for col in cat_columns:
            self.data[col] = self.data[col].astype('category')
        
        # Calculate memory savings
        memory_after = self.data.memory_usage(deep=True).sum() / (1024**2)
        savings_pct = (1 - memory_after / memory_before) * 100
        
        print(f"Memory usage reduced from {memory_before:.2f} MB to {memory_after:.2f} MB ({savings_pct:.1f}% savings)")
        self.metrics['memory_final'] = memory_after
        
        return self.data
    
    def encode_call_put(self):
        """Extract option type (call/put) from contract symbol and one-hot encode"""
        print("\nOne-hot encoding call/put option type...")
        
        if 'contractSymbol' in self.data.columns:
            # Extract C or P from contract symbol (format: AAPL220218C00070000)
            option_type = self.data['contractSymbol'].str.extract(r'(\d+)([CP])', expand=True)[1]
            
            # Create one-hot encoding
            self.data['is_call'] = (option_type == 'C').astype(int)
            self.data['is_put'] = (option_type == 'P').astype(int)
            
            print(f"  Found {self.data['is_call'].sum():,} calls and {self.data['is_put'].sum():,} puts")
        else:
            print("  'contractSymbol' not found, skipping call/put encoding")
        
        return self.data
    
    def encode_temporal_features(self):
        """
        Apply cyclical encoding to temporal features to capture seasonality.
        """
        print("\nEncoding temporal features...")
        
        # Ensure date columns are in datetime format
        if 'lastTradeDate' in self.data.columns and self.data['lastTradeDate'].dtype != 'datetime64[ns]':
            self.data['lastTradeDate'] = pd.to_datetime(self.data['lastTradeDate'])
        
        # Create day-of-week, day-of-month, and day-of-year
        if 'lastTradeDate' in self.data.columns:
            # Day of week (Monday=0, Sunday=6)
            self.data['day_of_week'] = self.data['lastTradeDate'].dt.dayofweek
            
            # Day of month (1..31)
            self.data['day_of_month'] = self.data['lastTradeDate'].dt.day
            
            # Day of year (1..365 or 366)
            self.data['day_of_year'] = self.data['lastTradeDate'].dt.dayofyear
            
            # Cyclical encoding
            # day_of_week => 0..6
            self.data['day_of_week_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
            self.data['day_of_week_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
            
            # day_of_month => 1..31
            self.data['day_of_month_sin'] = np.sin(2 * np.pi * (self.data['day_of_month'] - 1) / 31)
            self.data['day_of_month_cos'] = np.cos(2 * np.pi * (self.data['day_of_month'] - 1) / 31)
            
            # day_of_year => 1..365 (or 366)
            self.data['day_of_year_sin'] = np.sin(2 * np.pi * (self.data['day_of_year'] - 1) / 365)
            self.data['day_of_year_cos'] = np.cos(2 * np.pi * (self.data['day_of_year'] - 1) / 365)
            
            print("  Added cyclical time encodings for day of week, month, and year")
        else:
            print("  'lastTradeDate' not found, skipping temporal encoding")
        
        return self.data
    
    def load_risk_free_rates(self):
        """
        Load and prepare the risk-free rates from the DGS10 file
        Returns a dictionary mapping dates to rates
        """
        print("\nLoading risk-free rates from", self.rates_file)
        
        try:
            # Read the risk-free rates file
            rates_df = pd.read_csv(self.rates_file)
            
            # Check and rename columns if needed
            if 'observation_date' in rates_df.columns and 'DGS10' in rates_df.columns:
                # File has expected column structure
                pass
            elif len(rates_df.columns) == 2:
                # Assume the format is date and rate
                rates_df.columns = ['observation_date', 'DGS10']
            else:
                print(f"  Warning: Unexpected column structure in {self.rates_file}")
                print(f"  Columns found: {rates_df.columns.tolist()}")
                print("  Will try to use first two columns as date and rate")
                rates_df = rates_df.iloc[:, :2]
                rates_df.columns = ['observation_date', 'DGS10']
            
            # Convert date column to datetime
            rates_df['observation_date'] = pd.to_datetime(rates_df['observation_date'])
            
            # Convert rates to decimal (divide by 100 if they are in percentage form)
            if rates_df['DGS10'].mean() > 5:  # If rates are in percentage form (e.g., 1.85 means 1.85%)
                rates_df['DGS10'] = rates_df['DGS10'] / 100.0
            
            # Handle missing values by forward-filling
            rates_df = rates_df.sort_values('observation_date')
            rates_df['DGS10'] = rates_df['DGS10'].ffill()
            
            # Create a dictionary for quick lookup
            rates_dict = dict(zip(rates_df['observation_date'], rates_df['DGS10']))
            
            print(f"  Loaded {len(rates_dict)} daily risk-free rates")
            print(f"  Rate range: {min(rates_dict.values()):.4f} to {max(rates_dict.values()):.4f}")
            
            return rates_dict
            
        except Exception as e:
            print(f"  Error loading risk-free rates: {e}")
            print("  Falling back to constant risk-free rate of 0.02 (2%)")
            return None

    def feature_engineer_greeks(self):
        """
        Add additional options Greeks for model training.
        Uses Black-Scholes approximations with historical risk-free rates.
        """
        print("\nEngineering option Greeks...")
        
        # Check if we have the necessary columns
        required = ['is_call', 'strike', 'stockClose', 'impliedVolatility', 'daysToExpiry', 'quoteDate']
        if not all(col in self.data.columns for col in required):
            print("  Missing required columns for Greek calculations, skipping")
            return self.data
        
        # Load risk-free rates
        rates_dict = self.load_risk_free_rates()
        
        # If rates couldn't be loaded, use a constant rate
        if rates_dict is None:
            # Use a reasonable default rate
            r = 0.02  # 2% as a default
            self.data['risk_free_rate'] = r
            print("  Using constant risk-free rate:", r)
        else:
            # Map dates to rates
            print("  Mapping quote dates to risk-free rates...")
            
            # Initialize with default rate for dates not found
            default_rate = 0.02
            self.data['risk_free_rate'] = default_rate
            
            # Convert to datetime if not already
            if self.data['quoteDate'].dtype != 'datetime64[ns]':
                self.data['quoteDate'] = pd.to_datetime(self.data['quoteDate'])
            
            # Create a faster lookup by converting to period
            self.data['quote_date_key'] = self.data['quoteDate'].dt.floor('D')
            
            # Get unique dates for faster processing
            unique_dates = self.data['quote_date_key'].unique()
            rate_map = {}
            
            # For each unique date, find the closest available rate
            for date in unique_dates:
                # Try exact date match
                if date in rates_dict:
                    rate_map[date] = rates_dict[date]
                else:
                    # Find closest previous date
                    prev_dates = [d for d in rates_dict.keys() if d < date]
                    if prev_dates:
                        closest_date = max(prev_dates)
                        rate_map[date] = rates_dict[closest_date]
                    else:
                        # Use the first available rate if no previous date
                        first_date = min(rates_dict.keys())
                        rate_map[date] = rates_dict[first_date]
            
            # Map rates to data
            self.data['risk_free_rate'] = self.data['quote_date_key'].map(rate_map).fillna(default_rate)
            
            # Cleanup
            self.data = self.data.drop(columns=['quote_date_key'])
            
            # Verify mapping
            mapped_count = (self.data['risk_free_rate'] != default_rate).sum()
            print(f"  Mapped {mapped_count:,} of {len(self.data):,} rows to historical rates")
        
        # Assume zero dividend yield for simplicity (can be improved with actual dividend data)
        q = 0.0
        
        # Convert days to expiry to years
        self.data['time_to_expiry'] = self.data['daysToExpiry'] / 365.0
        
        # Filter out expired options or those with zero/negative time to expiry
        self.data = self.data[self.data['time_to_expiry'] > 0]
        
        from scipy.stats import norm
        
        # Calculate d1 with individual risk-free rates
        S = self.data['stockClose'].values
        K = self.data['strike'].values
        t = self.data['time_to_expiry'].values
        sigma = self.data['impliedVolatility'].values
        r = self.data['risk_free_rate'].values  # Now using per-row risk-free rates
        
        # Handle zeros and negatives to prevent warnings
        valid_mask = (sigma > 0) & (t > 0) & (S > 0) & (K > 0)
        
        # Initialize arrays
        d1 = np.zeros_like(S)
        d2 = np.zeros_like(S)
        
        # Calculate only for valid entries
        d1[valid_mask] = (np.log(S[valid_mask]/K[valid_mask]) + 
                          (r[valid_mask] - q + 0.5 * sigma[valid_mask]**2) * t[valid_mask]) / \
                         (sigma[valid_mask] * np.sqrt(t[valid_mask]))
        d2[valid_mask] = d1[valid_mask] - sigma[valid_mask] * np.sqrt(t[valid_mask])
        
        # Delta calculation
        self.data['delta'] = np.where(
            self.data['is_call'] == 1,
            np.exp(-q * t) * norm.cdf(d1),
            np.exp(-q * t) * (norm.cdf(d1) - 1)
        )
        
        # Gamma calculation (same for calls and puts)
        self.data['gamma'] = np.where(
            valid_mask,
            np.exp(-q * t[valid_mask]) * norm.pdf(d1[valid_mask]) / (S[valid_mask] * sigma[valid_mask] * np.sqrt(t[valid_mask])),
            0
        )
        
        # Theta calculation with vectorized r values
        def theta_call(S, K, t, sigma, r, d1, d2):
            return -S * np.exp(-q * t) * norm.pdf(d1) * sigma / (2 * np.sqrt(t)) - \
                   r * K * np.exp(-r * t) * norm.cdf(d2) + q * S * np.exp(-q * t) * norm.cdf(d1)
        
        def theta_put(S, K, t, sigma, r, d1, d2):
            return -S * np.exp(-q * t) * norm.pdf(d1) * sigma / (2 * np.sqrt(t)) + \
                   r * K * np.exp(-r * t) * norm.cdf(-d2) - q * S * np.exp(-q * t) * norm.cdf(-d1)
        
        # Create arrays for call and put thetas
        call_theta = np.zeros_like(S)
        put_theta = np.zeros_like(S)
        
        # Calculate only for valid entries to prevent warnings
        call_theta[valid_mask] = theta_call(
            S[valid_mask], K[valid_mask], t[valid_mask], 
            sigma[valid_mask], r[valid_mask], d1[valid_mask], d2[valid_mask]
        )
        
        put_theta[valid_mask] = theta_put(
            S[valid_mask], K[valid_mask], t[valid_mask], 
            sigma[valid_mask], r[valid_mask], d1[valid_mask], d2[valid_mask]
        )
        
        # Apply based on option type
        self.data['theta'] = np.where(
            self.data['is_call'] == 1,
            call_theta,
            put_theta
        )
        
        # Vega calculation (same for calls and puts)
        self.data['vega'] = S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t) / 100  # Divided by 100 for 1% change
        
        # Rho calculation with vectorized r values
        def rho_call(K, t, r, d2):
            return K * t * np.exp(-r * t) * norm.cdf(d2) / 100
        
        def rho_put(K, t, r, d2):
            return -K * t * np.exp(-r * t) * norm.cdf(-d2) / 100
        
        # Create arrays for call and put rhos
        call_rho = np.zeros_like(S)
        put_rho = np.zeros_like(S)
        
        # Calculate only for valid entries
        call_rho[valid_mask] = rho_call(
            K[valid_mask], t[valid_mask], r[valid_mask], d2[valid_mask]
        )
        
        put_rho[valid_mask] = rho_put(
            K[valid_mask], t[valid_mask], r[valid_mask], d2[valid_mask]
        )
        
        # Apply based on option type
        self.data['rho'] = np.where(
            self.data['is_call'] == 1,
            call_rho,
            put_rho
        )
        
        # Convert to float32 for consistency
        greek_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for col in greek_cols:
            self.data[col] = self.data[col].astype('float32')
        
        print(f"  Added {len(greek_cols)} Greeks to the dataset")
        
        return self.data
        
    def add_moneyness_feature(self):
        """
        Add moneyness (S/K) as a feature - useful for option pricing models
        """
        print("\nAdding moneyness feature...")
        
        if 'strike' in self.data.columns and 'stockClose' in self.data.columns:
            # Calculate moneyness as S/K
            self.data['moneyness'] = self.data['stockClose'] / self.data['strike']
            
            # For puts, we often want to use K/S
            self.data['moneyness_put'] = 1 / self.data['moneyness']
            
            # Calculate log-moneyness
            self.data['log_moneyness'] = np.log(self.data['moneyness'])
            
            # Convert to float32 for consistency
            self.data['moneyness'] = self.data['moneyness'].astype('float32')
            self.data['moneyness_put'] = self.data['moneyness_put'].astype('float32')
            self.data['log_moneyness'] = self.data['log_moneyness'].astype('float32')
            
            print(f"  Added moneyness features with range: {self.data['moneyness'].min():.2f} to {self.data['moneyness'].max():.2f}")
        else:
            print("  Cannot add moneyness - missing required columns")
        
        return self.data
        
    def verify_realistic_values(self):
        """
        Verify that the processed data has realistic values
        """
        print("\nVerifying realistic values...")
        
        issues_found = []
        
        # Check price columns
        price_cols = ['strike', 'bid', 'ask', 'stockClose']
        for col in price_cols:
            if col in self.data.columns and (self.data[col] < 0).any():
                neg_count = (self.data[col] < 0).sum()
                issues_found.append(f"{col} has {neg_count} negative values")
        
        # Check impliedVolatility
        if 'impliedVolatility' in self.data.columns:
            iv_max = self.data['impliedVolatility'].max()
            if iv_max > 5:  # 500% IV is extreme
                iv_high_count = (self.data['impliedVolatility'] > 5).sum()
                issues_found.append(f"impliedVolatility has {iv_high_count} values above 500%")
        
        # Check greeks for extreme values
        greek_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for col in greek_cols:
            if col in self.data.columns:
                if np.isnan(self.data[col]).any():
                    nan_count = np.isnan(self.data[col]).sum()
                    issues_found.append(f"{col} has {nan_count} NaN values")
                
                if np.isinf(self.data[col]).any():
                    inf_count = np.isinf(self.data[col]).sum()
                    issues_found.append(f"{col} has {inf_count} infinite values")
        
        # Replace NaN or inf values in Greeks
        for col in greek_cols:
            if col in self.data.columns:
                # Replace NaN or inf with 0
                mask = np.isnan(self.data[col]) | np.isinf(self.data[col])
                if mask.any():
                    self.data.loc[mask, col] = 0
                    print(f"  Replaced {mask.sum()} invalid values in {col} with 0")
        
        # Report issues
        if issues_found:
            print("  Issues found:")
            for issue in issues_found:
                print(f"  - {issue}")
        else:
            print("  ✓ All value checks passed")
        
        return self.data
        
    def validate_no_normalization(self):
        """
        Explicitly verify that no normalization has occurred by checking numeric ranges
        """
        print("\nVerifying data is not normalized...")
        
        # Check key numeric columns to ensure they maintain their original scale
        price_columns = ['strike', 'bid', 'ask', 'stockClose']
        price_columns = [col for col in price_columns if col in self.data.columns]
        
        # Sample the first 1000 rows to get stats
        sample = self.data.head(1000)
        
        for col in price_columns:
            col_min = sample[col].min()
            col_max = sample[col].max()
            col_mean = sample[col].mean()
            col_std = sample[col].std()
            
            # Check for signs of normalization
            is_normalized = (
                (col_min >= -3 and col_max <= 3 and col_std < 1.5) or  # Signs of standardization
                (col_min >= 0 and col_max <= 1 and col_std < 0.5)      # Signs of min-max scaling
            )
            
            if is_normalized:
                print(f"  WARNING: Column '{col}' appears to be normalized:")
                print(f"    Range: {col_min:.4f} to {col_max:.4f}, Mean: {col_mean:.4f}, Std: {col_std:.4f}")
                print("    This suggests data might have been unintentionally normalized!")
            else:
                print(f"  ✓ Column '{col}' is NOT normalized (range: {col_min:.2f} to {col_max:.2f})")
        
        # Check for features that might have been scaled
        if 'volume' in self.data.columns and self.data['volume'].max() <= 1:
            print("  WARNING: 'volume' column appears to be scaled down!")
        
        return not any(is_normalized for col in price_columns)
    
    def split_by_ticker(self):
        """Split the data by ticker and save to separate files"""
        print("\nSplitting data by ticker...")
        
        tickers = self.data['ticker'].unique()
        ticker_counts = {}
        
        print(f"Found {len(tickers)} unique tickers")
        
        for ticker in tickers:
            ticker_data = self.data[self.data['ticker'] == ticker]
            ticker_counts[ticker] = len(ticker_data)
            
            output_file = os.path.join(self.output_dir, f"option_data_scaled_{ticker}.csv")
            ticker_data.to_csv(output_file, index=False)
            
            print(f"  Saved {len(ticker_data):,} rows for {ticker} to {output_file}")
        
        # Save ticker metadata
        metadata = pd.DataFrame([
            {'ticker': ticker, 'count': count}
            for ticker, count in ticker_counts.items()
        ])
        
        metadata_file = os.path.join(self.output_dir, 'ticker_metadata.csv')
        metadata.to_csv(metadata_file, index=False)
        print(f"\nSaved ticker metadata to {metadata_file}")
        
        return ticker_counts
    
    def process_all(self):
        """Run the full preprocessing pipeline"""
        start_time = time.time()
        print(f"Started preprocessing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 1. Load data
        self.load_data()
        
        # 2. Analyze data quality and drop incomplete columns
        self.analyze_data_quality()
        self.drop_incomplete_columns()
        
        # 3. Clean data and drop rows with missing critical values
        self.clean_data()
        
        # 4. Filter data based on business rules
        self.filter_data()
        
        # 5. Apply stock splits
        self.apply_stock_splits()
        
        # 6. Validate stock split adjustments
        self.validate_stock_split_adjustments()
        
        # 7. Optimize datatypes
        self.optimize_datatypes()
        
        # 8. Encode call/put indicator
        self.encode_call_put()
        
        # 9. Encode temporal features
        self.encode_temporal_features()
        
        # 10. Feature engineer option Greeks
        self.feature_engineer_greeks()
        
        # 11. Add moneyness feature
        self.add_moneyness_feature()
        
        # 12. Verify realistic values
        self.verify_realistic_values()
        
        # 13. Verify no normalization was applied
        self.validate_no_normalization()
        
        # 14. Split by ticker and save
        self.split_by_ticker()
        
        # Print summary
        elapsed_min = (time.time() - start_time) / 60
        print("\n" + "=" * 80)
        print("Preprocessing Summary:")
        print(f"Initial rows: {self.metrics['rows_initial']:,}")
        print(f"Rows after cleaning: {self.metrics['rows_after_cleaning']:,}")
        print(f"Rows after filtering: {self.metrics['rows_after_filtering']:,}")
        print(f"Columns dropped: {self.metrics['columns_dropped']}")
        print(f"Memory reduced: {self.metrics['memory_initial']:.2f} MB → {self.metrics['memory_final']:.2f} MB " +
              f"({(1 - self.metrics['memory_final']/self.metrics['memory_initial'])*100:.1f}% reduction)")
        print(f"Total processing time: {elapsed_min:.2f} minutes")
        print("=" * 80)
        
        return self.data


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "reprocessing/DGS10.csv"  # Your NumPy data file
    RATES_FILE = "reprocessing/DGS10.csv"              # Risk-free rates file
    OUTPUT_DIR = "processed_data"
    
    # Run preprocessor
    preprocessor = OptionsPreprocessor(INPUT_FILE, RATES_FILE, OUTPUT_DIR)
    preprocessor.process_all()