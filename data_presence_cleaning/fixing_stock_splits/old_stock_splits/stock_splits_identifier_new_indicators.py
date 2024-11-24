import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

class StockSplitDetector:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.setup_logging()
        
        # Known splits during our period
        self.known_splits = {
            'TSLA': {'date': '2022-08-25', 'ratio': 1/3},  # 3:1 split
            'GOOGL': {'date': '2022-07-18', 'ratio': 1/20},  # 20:1 split
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def detect_split_indicators(self, df: pd.DataFrame) -> List[dict]:
        """
        Detect split indicators including:
        1. Strike price changes
        2. Volume spikes
        3. Number of options changes
        4. Price changes
        """
        splits = []
        dates = sorted(df['quoteDate'].unique())
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            # Get data for both dates
            prev_data = df[df['quoteDate'] == prev_date]
            curr_data = df[df['quoteDate'] == curr_date]
            
            indicators = 0  # Count how many indicators suggest a split
            
            # 1. Check strike price change
            prev_strike = prev_data['strike'].median()
            curr_strike = curr_data['strike'].median()
            if prev_strike > 0 and curr_strike > 0:
                strike_ratio = prev_strike / curr_strike
                if strike_ratio > 1.4 or strike_ratio < 0.7:
                    indicators += 1
            
            # 2. Check volume spike
            prev_volume = prev_data['volume'].sum()
            curr_volume = curr_data['volume'].sum()
            if curr_volume > prev_volume * 1.3:  # 30% increase in volume
                indicators += 1
            
            # 3. Check number of options
            prev_options = len(prev_data)
            curr_options = len(curr_data)
            if curr_options > prev_options * 1.3:  # 30% increase in options
                indicators += 1
            
            # 4. Check price change
            prev_price = prev_data['stockClose'].mean()
            curr_price = curr_data['stockClose'].mean()
            if prev_price > 0 and curr_price > 0:
                price_ratio = prev_price / curr_price
                if abs(1 - price_ratio) > 0.05:  # 5% price change
                    indicators += 1
            
            # If we have at least 3 indicators, consider it a split
            if indicators >= 3:
                splits.append({
                    'date': curr_date,
                    'indicators': indicators,
                    'strike_ratio': strike_ratio if 'strike_ratio' in locals() else None,
                    'volume_change': curr_volume / prev_volume if prev_volume > 0 else None,
                    'options_change': curr_options / prev_options if prev_options > 0 else None,
                    'price_ratio': price_ratio if 'price_ratio' in locals() else None
                })
        
        return splits

    def detect_splits(self) -> Dict[str, List[dict]]:
        """
        Detect stock splits using multiple indicators.
        """
        self.logger.info("Loading options data...")
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        splits_detected = {}
        unique_tickers = sorted(df['ticker'].unique())
        
        for ticker in tqdm(unique_tickers, desc="Analyzing tickers"):
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # Need enough data to detect splits
            if len(ticker_data) < 10:
                continue
                
            ticker_data = ticker_data.sort_values('quoteDate')
            
            # Detect splits
            potential_splits = self.detect_split_indicators(ticker_data)
            
            if potential_splits:
                # Consolidate nearby splits (within 3 days)
                consolidated = []
                current_split = potential_splits[0]
                
                for split in potential_splits[1:]:
                    date_diff = (split['date'] - current_split['date']).days
                    
                    if date_diff <= 3:
                        # Use the one with more indicators
                        if split['indicators'] > current_split['indicators']:
                            current_split = split
                    else:
                        consolidated.append(current_split)
                        current_split = split
                
                consolidated.append(current_split)
                
                # Calculate ratios and create final split records
                final_splits = []
                for split in consolidated:
                    ratio = None
                    if split['strike_ratio']:
                        ratio = split['strike_ratio']
                    elif split['price_ratio']:
                        ratio = split['price_ratio']
                    
                    if ratio:
                        final_splits.append({
                            'date': split['date'],
                            'ratio': min(ratio, 1/ratio),  # Always use fraction < 1
                            'indicators': {
                                'strike_ratio': split['strike_ratio'],
                                'volume_change': split['volume_change'],
                                'options_change': split['options_change'],
                                'price_ratio': split['price_ratio']
                            }
                        })
                
                if final_splits:
                    splits_detected[ticker] = final_splits
                    
                    self.logger.info(f"\nConfirmed splits for {ticker}:")
                    for split in final_splits:
                        ratio = split['ratio']
                        split_text = f"{1/ratio:.1f}:1 split"
                        self.logger.info(
                            f"Date: {split['date'].strftime('%Y-%m-%d')}, {split_text}")
                        self.logger.info("Indicators:")
                        for name, value in split['indicators'].items():
                            if value:
                                self.logger.info(f"  {name}: {value:.2f}")
        
        # Verify against known splits
        self.logger.info("\nKnown splits verification:")
        for ticker, known_split in self.known_splits.items():
            if ticker in splits_detected:
                self.logger.info(f"{ticker}: ✓ Detected")
                # Verify the date
                known_date = pd.to_datetime(known_split['date'])
                detected_dates = [s['date'] for s in splits_detected[ticker]]
                closest_date = min(detected_dates, key=lambda d: abs((d - known_date).days))
                date_diff = abs((closest_date - known_date).days)
                if date_diff > 0:
                    self.logger.warning(
                        f"  Date difference: {date_diff} days "
                        f"(detected: {closest_date.date()}, actual: {known_date.date()})"
                    )
            else:
                self.logger.warning(f"{ticker}: ✗ Not detected")
        
        return splits_detected

if __name__ == "__main__":
    detector = StockSplitDetector('data_files/option_data_no_BACC_adjusted.csv')
    splits = detector.detect_splits()