import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging

class StockSplitAdjuster:
    def __init__(self, data_path: str, output_path: str = None):
        """Initialize the split adjuster."""
        self.data_path = data_path
        self.output_path = output_path or data_path.replace('.csv', '_adjusted.csv')
        self.setup_logging()
        
        # Features that need direct split adjustment (divide by split ratio)
        self.price_features = [
            'strike',
            'lastPrice',
            'bid',
            'ask',
            'change',
            'stockClose',
            'stockOpen',
            'stockHigh',
            'stockLow',
            'stockClose_ewm_5d',
            'stockClose_ewm_15d',
            'stockClose_ewm_45d',
            'stockClose_ewm_135d',
            'strikeDelta'
        ]
        
        # Features that need inverse split adjustment (multiply by split ratio)
        self.volume_features = [
            'volume',
            'openInterest',
            'stockVolume'
        ]
        
        # Features that don't need adjustment
        self.no_adjustment_features = [
            'contractSymbol',
            'lastTradeDate',
            'percentChange',
            'impliedVolatility',
            'inTheMoney',
            'contractSize',
            'currency',
            'quoteDate',
            'expiryDate',
            'daysToExpiry',
            'ticker',
            'stockAdjClose'  # Already adjusted for splits
        ]

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def detect_splits(self, df: pd.DataFrame) -> Dict[str, List[dict]]:
        """
        Detect stock splits using stockAdjClose and multiple confirmatory indicators.
        """
        splits_detected = {}
        
        for ticker in tqdm(df['ticker'].unique(), desc="Detecting splits"):
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('quoteDate')
            
            # Compare stockClose to stockAdjClose to detect splits
            ticker_data['close_ratio'] = ticker_data['stockClose'] / ticker_data['stockAdjClose']
            
            # Group by date to get daily metrics
            daily_data = ticker_data.groupby('quoteDate').agg({
                'close_ratio': 'mean',
                'strike': 'median',
                'volume': 'sum',
                'stockClose': 'mean',
                'stockAdjClose': 'mean'
            }).reset_index()
            
            potential_splits = []
            
            for i in range(1, len(daily_data)):
                indicators = 0
                metrics = {}
                
                # 1. Check close to adjusted close ratio change
                ratio_change = daily_data.iloc[i]['close_ratio'] / daily_data.iloc[i-1]['close_ratio']
                if abs(1 - ratio_change) > 0.1:  # 10% change threshold
                    indicators += 1
                    metrics['ratio_change'] = ratio_change
                
                # 2. Check strike price changes
                strike_ratio = daily_data.iloc[i-1]['strike'] / daily_data.iloc[i]['strike']
                if abs(1 - strike_ratio) > 0.4:  # 40% change threshold
                    indicators += 1
                    metrics['strike_ratio'] = strike_ratio
                
                # 3. Check volume spikes
                volume_ratio = (daily_data.iloc[i]['volume'] / 
                              daily_data.iloc[i-1]['volume'])
                if volume_ratio > 1.3:  # 30% increase
                    indicators += 1
                    metrics['volume_ratio'] = volume_ratio
                
                # If we have enough indicators
                if indicators >= 2:
                    # Use stockClose/stockAdjClose ratio to determine split ratio
                    split_ratio = daily_data.iloc[i-1]['stockClose'] / daily_data.iloc[i-1]['stockAdjClose']
                    
                    # Round to nearest common split ratio
                    common_ratios = [2, 3, 4, 5, 10, 20]
                    ratio = min(common_ratios, key=lambda x: abs(split_ratio - x))
                    
                    potential_splits.append({
                        'date': daily_data.iloc[i]['quoteDate'],
                        'ratio': ratio,
                        'metrics': metrics
                    })
            
            if potential_splits:
                # Consolidate nearby splits (within 5 days)
                consolidated = []
                current_split = potential_splits[0]
                
                for split in potential_splits[1:]:
                    date_diff = (split['date'] - current_split['date']).days
                    
                    if date_diff <= 5:
                        # Keep the split with more confirming metrics
                        if len(split['metrics']) > len(current_split['metrics']):
                            current_split = split
                    else:
                        consolidated.append(current_split)
                        current_split = split
                
                consolidated.append(current_split)
                splits_detected[ticker] = consolidated
        
        return splits_detected

    def adjust_features(self, df: pd.DataFrame, splits: Dict[str, List[dict]]) -> pd.DataFrame:
        """Adjust all relevant features for splits."""
        df_adjusted = df.copy()
        splits_applied = []
        
        for ticker, ticker_splits in splits.items():
            ticker_mask = df_adjusted['ticker'] == ticker
            
            for split in ticker_splits:
                split_date = split['date']
                split_ratio = split['ratio']
                
                # Mask for data before split
                before_split = ticker_mask & (df_adjusted['quoteDate'] < split_date)
                
                # Price adjustments (divide by split ratio)
                for feature in self.price_features:
                    if feature in df_adjusted.columns:
                        # Skip adjustment if column is all zeros or nulls
                        if df_adjusted[feature].abs().max() > 0:
                            df_adjusted.loc[before_split, feature] /= split_ratio
                
                # Volume adjustments (multiply by split ratio)
                for feature in self.volume_features:
                    if feature in df_adjusted.columns:
                        # Skip adjustment if column is all zeros or nulls
                        if df_adjusted[feature].abs().max() > 0:
                            df_adjusted.loc[before_split, feature] *= split_ratio
                
                splits_applied.append({
                    'ticker': ticker,
                    'date': split_date,
                    'ratio': split_ratio,
                    'records_adjusted': before_split.sum()
                })
                
                self.logger.info(
                    f"Adjusted {ticker} for {split_ratio}:1 split on {split_date.date()} "
                    f"({before_split.sum():,} records)"
                )
        
        return df_adjusted, splits_applied

    def process(self) -> None:
        """Main processing function."""
        # Load data
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        # Detect splits
        splits = self.detect_splits(df)
        
        # Apply adjustments
        df_adjusted, splits_applied = self.adjust_features(df, splits)
        
        # Save adjusted data
        self.logger.info(f"\nSaving adjusted data to {self.output_path}")
        df_adjusted.to_csv(self.output_path, index=False)
        
        # Log summary
        self.logger.info("\nAdjustment Summary:")
        self.logger.info(f"Total records processed: {len(df):,}")
        self.logger.info(f"Splits applied: {len(splits_applied)}")
        self.logger.info("\nAdjusted price features:")
        for feature in self.price_features:
            if feature in df.columns:
                self.logger.info(f"  - {feature}")
        self.logger.info("\nAdjusted volume features:")
        for feature in self.volume_features:
            if feature in df.columns:
                self.logger.info(f"  - {feature}")

if __name__ == "__main__":
    adjuster = StockSplitAdjuster('data_files/option_data_no_BACC.csv')
    adjuster.process()