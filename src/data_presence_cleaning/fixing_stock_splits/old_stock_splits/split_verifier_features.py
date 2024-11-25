import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging

class SplitVerifier:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.setup_logging()
        
        # Known splits with exact ratios and dates
        self.verified_splits = {
            'AMZN': {
                'date': '2022-06-06',
                'ratio': 20.0,
                'type': 'forward'
            },
            'GOOGL': {
                'date': '2022-07-18',
                'ratio': 20.0,
                'type': 'forward'
            },
            'TSLA': {
                'date': '2022-08-25',
                'ratio': 3.0,
                'type': 'forward'
            },
            'SHOP': {
                'date': '2022-07-04',
                'ratio': 10.0,
                'type': 'forward'
            },
            'CGC': {
                'date': '2023-12-20',
                'ratio': 10.0,  # 1:10 reverse
                'type': 'reverse'
            }
        }
        
        # Features to analyze
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

    def analyze_feature_around_split(
        self, 
        df: pd.DataFrame,
        ticker: str,
        feature: str,
        split_date: pd.Timestamp,
        split_ratio: float,
        window_days: int = 3
    ) -> Dict:
        """Analyze a single feature around the split date."""
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Get data before and after split
        pre_split = ticker_data[
            (ticker_data['quoteDate'] < split_date) & 
            (ticker_data['quoteDate'] >= split_date - timedelta(days=window_days))
        ]
        
        post_split = ticker_data[
            (ticker_data['quoteDate'] > split_date) & 
            (ticker_data['quoteDate'] <= split_date + timedelta(days=window_days))
        ]
        
        if len(pre_split) == 0 or len(post_split) == 0:
            return None
            
        # Calculate various metrics
        pre_stats = {
            'median': pre_split[feature].median(),
            'mean': pre_split[feature].mean(),
            'std': pre_split[feature].std(),
            'min': pre_split[feature].min(),
            'max': pre_split[feature].max(),
            'count': len(pre_split)
        }
        
        post_stats = {
            'median': post_split[feature].median(),
            'mean': post_split[feature].mean(),
            'std': post_split[feature].std(),
            'min': post_split[feature].min(),
            'max': post_split[feature].max(),
            'count': len(post_split)
        }
        
        # Calculate ratios and compare to expected
        observed_ratio = post_stats['median'] / pre_stats['median'] if pre_stats['median'] != 0 else np.nan
        expected_ratio = split_ratio
        ratio_diff = abs(1 - (observed_ratio / expected_ratio)) if not np.isnan(observed_ratio) else np.nan
        
        # Check against stockAdjClose if available
        if 'stockAdjClose' in df.columns:
            adj_close_pre = pre_split['stockAdjClose'].median()
            adj_close_post = post_split['stockAdjClose'].median()
            adj_close_ratio = adj_close_post / adj_close_pre if adj_close_pre != 0 else np.nan
        else:
            adj_close_ratio = np.nan
        
        return {
            'pre_split': pre_stats,
            'post_split': post_stats,
            'observed_ratio': observed_ratio,
            'expected_ratio': expected_ratio,
            'ratio_diff': ratio_diff,
            'adj_close_ratio': adj_close_ratio
        }

    def verify_splits(self) -> Dict:
        """Verify all splits and analyze features."""
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        # Analyze each split
        results = {}
        
        for ticker, split_info in self.verified_splits.items():
            split_date = pd.to_datetime(split_info['date'])
            split_ratio = split_info['ratio']
            split_type = split_info['type']
            
            self.logger.info(f"\nAnalyzing {ticker} {split_type} split on {split_date.date()}")
            self.logger.info(f"Ratio: {'1:' if split_type == 'reverse' else ''}{split_ratio}{'1' if split_type == 'forward' else ''}")
            
            # Analyze each feature
            feature_results = {}
            
            # Price features
            self.logger.info("\nPrice Features:")
            for feature in self.price_features:
                if feature in df.columns:
                    analysis = self.analyze_feature_around_split(
                        df, ticker, feature, split_date, split_ratio
                    )
                    if analysis:
                        feature_results[feature] = analysis
                        self.logger.info(f"\n{feature}:")
                        self.logger.info(f"  Pre-split  median: {analysis['pre_split']['median']:>10.2f}")
                        self.logger.info(f"  Post-split median: {analysis['post_split']['median']:>10.2f}")
                        self.logger.info(f"  Observed ratio:    {analysis['observed_ratio']:>10.2f}")
                        self.logger.info(f"  Expected ratio:    {analysis['expected_ratio']:>10.2f}")
                        self.logger.info(f"  Ratio difference:  {analysis['ratio_diff']:>10.2%}")
                        if not np.isnan(analysis['adj_close_ratio']):
                            self.logger.info(f"  AdjClose ratio:    {analysis['adj_close_ratio']:>10.2f}")
            
            # Volume features
            self.logger.info("\nVolume Features:")
            for feature in self.volume_features:
                if feature in df.columns:
                    analysis = self.analyze_feature_around_split(
                        df, ticker, feature, split_date, 1/split_ratio  # Inverse ratio for volumes
                    )
                    if analysis:
                        feature_results[feature] = analysis
                        self.logger.info(f"\n{feature}:")
                        self.logger.info(f"  Pre-split  median: {analysis['pre_split']['median']:>10.2f}")
                        self.logger.info(f"  Post-split median: {analysis['post_split']['median']:>10.2f}")
                        self.logger.info(f"  Observed ratio:    {analysis['observed_ratio']:>10.2f}")
                        self.logger.info(f"  Expected ratio:    {analysis['expected_ratio']:>10.2f}")
                        self.logger.info(f"  Ratio difference:  {analysis['ratio_diff']:>10.2%}")
            
            results[ticker] = feature_results
            
            # Summary recommendations
            self.logger.info("\nRecommendations:")
            for feature, analysis in feature_results.items():
                if analysis['ratio_diff'] > 0.1:  # More than 10% difference
                    if feature in self.price_features:
                        if analysis['ratio_diff'] > 0.9:  # Likely already adjusted
                            self.logger.info(f"  {feature}: Appears to be already adjusted")
                        else:
                            self.logger.info(f"  {feature}: Needs price adjustment")
                    else:  # Volume features
                        if analysis['ratio_diff'] > 0.9:  # Likely already adjusted
                            self.logger.info(f"  {feature}: Appears to be already adjusted")
                        else:
                            self.logger.info(f"  {feature}: Needs volume adjustment")
        
        return results

if __name__ == "__main__":
    verifier = SplitVerifier('data_files/option_data_no_BACC.csv')
    verification_results = verifier.verify_splits()