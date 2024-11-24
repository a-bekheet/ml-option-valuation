import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

class StockSplitDetector:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.setup_logging()
        
        # Known splits during our period for validation
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

    def analyze_price_series(self, series: pd.Series) -> List[dict]:
        """
        Analyze a price series for potential splits.
        """
        splits = []
        dates = series.index
        values = series.values
        
        for i in range(1, len(values)):
            # Calculate daily ratio
            ratio = values[i] / values[i-1] if values[i-1] != 0 else 1
            
            # Check for significant changes that might indicate splits
            if ratio < 0.4 or ratio > 2.5:  # Detect both splits and reverse splits
                # Look for clean ratios (e.g., 0.5 for 2:1 split, 0.333 for 3:1 split)
                common_ratios = [0.5, 0.333, 0.25, 0.2, 0.125]  # 2:1, 3:1, 4:1, 5:1, 8:1 splits
                for common_ratio in common_ratios:
                    if abs(ratio - common_ratio) < 0.05:  # 5% tolerance
                        splits.append({
                            'date': dates[i],
                            'ratio': ratio,
                            'before_price': values[i-1],
                            'after_price': values[i]
                        })
                        break
        
        return splits

    def detect_splits(self) -> Dict[str, List[dict]]:
        """
        Detect stock splits using multiple validation methods.
        """
        self.logger.info("Loading options data...")
        
        # Read data
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        splits_detected = {}
        unique_tickers = sorted(df['ticker'].unique())  # Sort tickers for consistent ordering
        
        for ticker in tqdm(unique_tickers, desc="Analyzing tickers"):
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('quoteDate')
            
            potential_splits = []
            
            # Method 1: Stock price analysis
            daily_prices = ticker_data.groupby('quoteDate')['stockClose'].mean()
            stock_splits = self.analyze_price_series(daily_prices)
            
            for split in stock_splits:
                potential_splits.append({
                    **split,
                    'method': 'stock_price'
                })
            
            # Method 2: Strike price analysis
            for expiry in ticker_data['expiryDate'].unique():
                expiry_data = ticker_data[ticker_data['expiryDate'] == expiry]
                if len(expiry_data) > 1:  # Need at least 2 points
                    daily_strikes = expiry_data.groupby('quoteDate')['strike'].mean()
                    if len(daily_strikes) > 1:  # Double-check after groupby
                        strike_splits = self.analyze_price_series(daily_strikes)
                        
                        for split in strike_splits:
                            potential_splits.append({
                                **split,
                                'method': 'strike_price'
                            })
            
            # Method 3: AdjClose comparison
            if 'stockAdjClose' in ticker_data.columns:
                daily_adj = ticker_data.groupby('quoteDate').agg({
                    'stockClose': 'mean',
                    'stockAdjClose': 'mean'
                })
                
                adj_ratio = daily_adj['stockClose'] / daily_adj['stockAdjClose']
                
                for date, ratio in adj_ratio.items():
                    if ratio > 1.5 or ratio < 0.67:  # Significant difference
                        common_ratios = [2, 3, 4, 5, 8]  # Common split ratios
                        for common in common_ratios:
                            if abs(ratio - common) < 0.2:  # 20% tolerance
                                potential_splits.append({
                                    'date': date,
                                    'ratio': 1/ratio,  # Convert to split ratio
                                    'method': 'adj_close'
                                })
            
            # Consolidate splits
            if potential_splits:
                # Sort by date
                potential_splits.sort(key=lambda x: x['date'])
                
                # Group nearby dates (within 5 business days)
                consolidated = []
                current_group = [potential_splits[0]]
                
                for split in potential_splits[1:]:
                    last_date = current_group[-1]['date']
                    current_date = split['date']
                    
                    if abs((current_date - last_date).days) <= 5:
                        current_group.append(split)
                    else:
                        # Process current group if it has multiple confirmations
                        methods = set(s['method'] for s in current_group)
                        if len(methods) >= 2:  # Require at least 2 methods to confirm
                            ratios = [s['ratio'] for s in current_group]
                            avg_ratio = np.median(ratios)  # Use median for robustness
                            consolidated.append({
                                'date': current_group[0]['date'],
                                'ratio': avg_ratio,
                                'methods': list(methods)
                            })
                        current_group = [split]
                
                # Don't forget the last group
                methods = set(s['method'] for s in current_group)
                if len(methods) >= 2:
                    ratios = [s['ratio'] for s in current_group]
                    avg_ratio = np.median(ratios)
                    consolidated.append({
                        'date': current_group[0]['date'],
                        'ratio': avg_ratio,
                        'methods': list(methods)
                    })
                
                if consolidated:
                    splits_detected[ticker] = consolidated
                    
                    self.logger.info(f"\nConfirmed splits for {ticker}:")
                    for split in consolidated:
                        ratio = split['ratio']
                        split_text = f"{1/ratio:.1f}:1 split" if ratio < 1 else f"1:{ratio:.1f} reverse split"
                        self.logger.info(
                            f"Date: {split['date'].strftime('%Y-%m-%d')}, "
                            f"{split_text}, "
                            f"Detected by: {', '.join(split['methods'])}"
                        )
        
        if splits_detected:
            self.logger.info("\nSummary of detected splits:")
            for ticker, split_list in splits_detected.items():
                for split in split_list:
                    ratio = split['ratio']
                    split_text = f"{1/ratio:.1f}:1" if ratio < 1 else f"1:{ratio:.1f}"
                    self.logger.info(
                        f"{ticker}: {split['date'].strftime('%Y-%m-%d')} - "
                        f"{split_text} split"
                    )
        else:
            self.logger.warning("No splits detected!")
            
        # Verify against known splits
        self.logger.info("\nKnown splits verification:")
        for ticker, known_split in self.known_splits.items():
            if ticker in splits_detected:
                self.logger.info(f"{ticker}: ✓ Detected")
            else:
                self.logger.warning(f"{ticker}: ✗ Not detected")
        
        return splits_detected

if __name__ == "__main__":
    detector = StockSplitDetector('data_files/option_data_no_BACC.csv')
    splits = detector.detect_splits()