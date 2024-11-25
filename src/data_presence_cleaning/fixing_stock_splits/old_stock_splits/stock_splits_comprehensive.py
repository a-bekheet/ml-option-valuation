import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging

class StockSplitDetector:
    def __init__(self, data_path: str, output_path: str = 'suspected_stock_splits.csv'):
        self.data_path = data_path
        self.output_path = output_path
        self.setup_logging()
        
        # Known verified splits
        self.verified_splits = {
            'AMZN': {'date': '2022-06-06', 'ratio': 20.0},
            'GOOGL': {'date': '2022-07-18', 'ratio': 20.0},
            'TSLA': {'date': '2022-08-25', 'ratio': 3.0},
            'SHOP': {'date': '2022-07-04', 'ratio': 10.0},
            'CGC': {'date': '2023-12-20', 'ratio': 0.1}  # 1:10 reverse split
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def is_valid_split_ratio(self, ratio: float) -> bool:
        """Check if ratio matches common split ratios with wider tolerance."""
        common_ratios = [0.1, 0.2, 0.25, 0.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
        tolerance = 0.25  # Increased tolerance to 25%
        
        return any(abs(1 - (ratio / r)) < tolerance for r in common_ratios)

    def detect_splits(self) -> pd.DataFrame:
        """Enhanced split detection with focus on verified splits."""
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])

        all_splits = []
        detected_verified_splits = set()

        # Process each ticker
        for ticker in tqdm(df['ticker'].unique(), desc="Analyzing tickers"):
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('quoteDate')
            
            # Get daily metrics
            daily_data = ticker_data.groupby('quoteDate').agg({
                'strike': 'median',
                'stockClose': 'mean',
                'stockAdjClose': 'mean',
                'volume': 'sum',
                'stockVolume': 'sum',
                'openInterest': 'sum'
            }).reset_index()
            
            daily_data['num_options'] = ticker_data.groupby('quoteDate').size()
            
            # Check for splits
            for i in range(1, len(daily_data)):
                curr_date = daily_data.iloc[i]['quoteDate']
                prev_date = daily_data.iloc[i-1]['quoteDate']
                
                # Skip if dates too far apart
                if (curr_date - prev_date).days > 7:  # Increased window to 7 days
                    continue
                
                # Calculate ratios
                ratios = {
                    'strike': daily_data.iloc[i-1]['strike'] / daily_data.iloc[i]['strike'],
                    'price': daily_data.iloc[i-1]['stockClose'] / daily_data.iloc[i]['stockClose'],
                    'adj_price': daily_data.iloc[i-1]['stockAdjClose'] / daily_data.iloc[i]['stockAdjClose'],
                    'volume': daily_data.iloc[i]['volume'] / daily_data.iloc[i-1]['volume'],
                    'options': daily_data.iloc[i]['num_options'] / daily_data.iloc[i-1]['num_options']
                }
                
                # Clean any inf/nan values
                ratios = {k: v for k, v in ratios.items() if np.isfinite(v)}
                
                # Check for potential split
                is_split = False
                split_ratio = None
                
                # First check strike ratio
                if 'strike' in ratios and self.is_valid_split_ratio(ratios['strike']):
                    split_ratio = ratios['strike']
                    is_split = True
                
                # Then check price ratio if strike ratio didn't work
                elif 'price' in ratios and self.is_valid_split_ratio(ratios['price']):
                    split_ratio = ratios['price']
                    is_split = True
                
                # Additional confirmatory checks
                if is_split:
                    # Volume or options increase
                    volume_spike = ratios.get('volume', 1.0) > 1.2  # Reduced threshold
                    options_spike = ratios.get('options', 1.0) > 1.2
                    
                    # Context data
                    window_start = max(0, i-2)
                    window_end = min(len(daily_data), i+3)
                    context = daily_data.iloc[window_start:window_end][
                        ['quoteDate', 'stockClose', 'stockAdjClose', 'strike']
                    ].to_dict('records')
                    
                    if volume_spike or options_spike:
                        split_info = {
                            'ticker': ticker,
                            'date': curr_date,
                            'split_ratio': split_ratio,
                            'type': 'reverse' if split_ratio < 1 else 'regular',
                            'metrics': ratios,
                            'context': context
                        }
                        all_splits.append(split_info)
                        
                        # Check if this matches a verified split
                        if ticker in self.verified_splits:
                            verified_date = pd.to_datetime(self.verified_splits[ticker]['date'])
                            if abs((curr_date - verified_date).days) <= 7:  # Increased window
                                detected_verified_splits.add(ticker)
        
        # Create DataFrame
        if all_splits:
            splits_df = pd.DataFrame(all_splits)
            splits_df['metrics'] = splits_df['metrics'].apply(str)
            splits_df['context'] = splits_df['context'].apply(str)
            splits_df = splits_df.sort_values(['ticker', 'date'])
            
            # Save to CSV
            splits_df.to_csv(self.output_path, index=False)
            
            # Log validation results
            self.logger.info("\nValidation against known splits:")
            for ticker, split in self.verified_splits.items():
                status = "✓ Detected" if ticker in detected_verified_splits else "✗ Not detected"
                self.logger.info(f"{ticker} ({split['date']}): {status}")
            
            return splits_df
        else:
            self.logger.warning("No splits detected!")
            return pd.DataFrame()

if __name__ == "__main__":
    detector = StockSplitDetector('data_files/option_data_no_BACC.csv')
    suspected_splits = detector.detect_splits()