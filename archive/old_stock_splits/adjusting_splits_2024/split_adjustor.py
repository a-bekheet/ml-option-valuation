import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from tqdm import tqdm
import logging

class StockSplitAdjuster:
    def __init__(self, data_path: str, output_path: str = None):
        self.data_path = data_path
        self.output_path = output_path or data_path.replace('.csv', '_split_adjusted.csv')
        self.setup_logging()
        
        # Manually verified splits with exact ratios
        self.verified_splits = {
            'AMZN': {
                'date': '2022-06-06',
                'ratio': 20.0,  # 20:1 split
                'type': 'forward'
            },
            'GOOGL': {
                'date': '2022-07-18',
                'ratio': 20.0,  # 20:1 split
                'type': 'forward'
            },
            'TSLA': {
                'date': '2022-08-25',
                'ratio': 3.0,   # 3:1 split
                'type': 'forward'
            },
            'SHOP': {
                'date': '2022-07-04',
                'ratio': 10.0,  # 10:1 split
                'type': 'forward'
            },
            'CGC': {
                'date': '2023-12-20',
                'ratio': 10.0,  # 1:10 reverse split (stored as multiplier)
                'type': 'reverse'
            }
        }
        
        self.price_features = [
            'strike', 'lastPrice', 'bid', 'ask', 'change',
            'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
            'strikeDelta', 'stockClose_ewm_5d', 'stockClose_ewm_15d',
            'stockClose_ewm_45d', 'stockClose_ewm_135d'
        ]
        
        self.volume_features = [
            'volume', 'openInterest', 'stockVolume'
        ]

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def verify_adjustment(self, df: pd.DataFrame, ticker: str, split_date: pd.Timestamp, 
                         window_days: int = 3) -> None:
        """Verify the adjustment using a smaller window around the split."""
        ticker_data = df[df['ticker'] == ticker].copy()
        
        pre_split = ticker_data[
            (ticker_data['quoteDate'] < split_date) & 
            (ticker_data['quoteDate'] >= split_date - timedelta(days=window_days))
        ]
        
        post_split = ticker_data[
            (ticker_data['quoteDate'] > split_date) & 
            (ticker_data['quoteDate'] <= split_date + timedelta(days=window_days))
        ]
        
        if len(pre_split) == 0 or len(post_split) == 0:
            self.logger.warning(f"Insufficient data to verify {ticker} split")
            return
        
        # Check price continuity using median to avoid outliers
        pre_price = pre_split['stockClose'].median()
        post_price = post_split['stockClose'].median()
        if pre_price > 0 and post_price > 0:
            price_disc = abs(1 - (pre_price / post_price))
            self.logger.info(f"  Price continuity check:")
            self.logger.info(f"    Pre-split median price: {pre_price:.2f}")
            self.logger.info(f"    Post-split median price: {post_price:.2f}")
            self.logger.info(f"    Discontinuity: {price_disc:.1%}")
        
        # Check strike price continuity
        pre_strike = pre_split['strike'].median()
        post_strike = post_split['strike'].median()
        if pre_strike > 0 and post_strike > 0:
            strike_disc = abs(1 - (pre_strike / post_strike))
            self.logger.info(f"  Strike price continuity check:")
            self.logger.info(f"    Pre-split median strike: {pre_strike:.2f}")
            self.logger.info(f"    Post-split median strike: {post_strike:.2f}")
            self.logger.info(f"    Discontinuity: {strike_disc:.1%}")

    def adjust_data(self) -> pd.DataFrame:
        """Adjust the data for stock splits with improved handling."""
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        # Create copy for adjustment
        df_adjusted = df.copy()
        
        # Convert numeric columns to float64
        for col in self.price_features + self.volume_features:
            if col in df_adjusted.columns:
                df_adjusted[col] = df_adjusted[col].astype('float64')
        
        # Process each split
        self.logger.info("\nAdjusting for splits:")
        for ticker, split_info in self.verified_splits.items():
            split_date = pd.to_datetime(split_info['date'])
            split_ratio = split_info['ratio']
            split_type = split_info['type']
            
            # Get pre-split data mask
            ticker_mask = df_adjusted['ticker'] == ticker
            pre_split_mask = ticker_mask & (df_adjusted['quoteDate'] < split_date)
            affected_records = pre_split_mask.sum()
            
            self.logger.info(f"\nProcessing {ticker} {split_type} split on {split_date.date()}")
            if split_type == 'forward':
                self.logger.info(f"Ratio: {split_ratio}:1")
            else:  # reverse split
                self.logger.info(f"Ratio: 1:{split_ratio}")
            self.logger.info(f"Affected records: {affected_records:,}")
            
            if affected_records == 0:
                self.logger.warning(f"No pre-split records found for {ticker}")
                continue
            
            # Apply adjustments based on split type
            adjustment_ratio = split_ratio if split_type == 'forward' else (1 / split_ratio)
            
            # Adjust price features
            for feature in self.price_features:
                if feature in df_adjusted.columns:
                    mask = pre_split_mask & df_adjusted[feature].notna()
                    df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] / adjustment_ratio
            
            # Adjust volume features (inverse adjustment)
            for feature in self.volume_features:
                if feature in df_adjusted.columns:
                    mask = pre_split_mask & df_adjusted[feature].notna()
                    df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] * adjustment_ratio
            
            # Verify adjustments
            self.verify_adjustment(df_adjusted, ticker, split_date)
        
        # Save adjusted data
        self.logger.info(f"\nSaving adjusted data to {self.output_path}")
        df_adjusted.to_csv(self.output_path, index=False)
        
        return df_adjusted

if __name__ == "__main__":
    adjuster = StockSplitAdjuster('data_files/option_data_no_BACC.csv')
    adjusted_data = adjuster.adjust_data()