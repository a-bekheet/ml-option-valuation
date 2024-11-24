import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging

class FinalSplitDetector:
    def __init__(self, data_path: str, output_path: str = 'final_suspected_splits.csv'):
        self.data_path = data_path
        self.output_path = output_path
        self.setup_logging()
        
        # Known verified splits for validation
        self.verified_splits = {
            'AMZN': {'date': '2022-06-06', 'ratio': 20.0, 'type': 'split'},
            'GOOGL': {'date': '2022-07-18', 'ratio': 20.0, 'type': 'split'},
            'TSLA': {'date': '2022-08-25', 'ratio': 3.0, 'type': 'split'},
            'SHOP': {'date': '2022-07-04', 'ratio': 10.0, 'type': 'split'},
            'CGC': {'date': '2023-12-20', 'ratio': 0.1, 'type': 'reverse'}
        }
        
        # Common split ratios (both regular and reverse)
        self.common_ratios = {
            20.0: "20:1 split",
            10.0: "10:1 split",
            5.0: "5:1 split",
            4.0: "4:1 split",
            3.0: "3:1 split",
            2.0: "2:1 split",
            0.5: "1:2 reverse",
            0.333: "1:3 reverse",
            0.25: "1:4 reverse",
            0.2: "1:5 reverse",
            0.1: "1:10 reverse",
            0.05: "1:20 reverse"
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def identify_split_ratio(self, observed_ratio: float) -> Tuple[float, str]:
        """Identify the closest common split ratio."""
        if not np.isfinite(observed_ratio) or observed_ratio <= 0:
            return None, None
            
        # Find closest ratio
        closest_ratio = min(self.common_ratios.keys(), 
                          key=lambda x: abs(np.log(observed_ratio) - np.log(x)))
        
        # Check if within tolerance (25%)
        if abs(1 - (observed_ratio / closest_ratio)) < 0.25:
            return closest_ratio, self.common_ratios[closest_ratio]
        return None, None

    def analyze_ticker_data(self, ticker: str, data: pd.DataFrame) -> List[dict]:
        """Analyze a single ticker for potential splits."""
        data = data.sort_values('quoteDate')
        
        # Get daily metrics
        daily_data = data.groupby('quoteDate').agg({
            'strike': 'median',
            'stockClose': 'mean',
            'stockAdjClose': 'mean',
            'volume': 'sum',
            'stockVolume': 'sum',
            'openInterest': 'sum'
        }).reset_index()
        
        daily_data['num_options'] = data.groupby('quoteDate').size()
        
        potential_splits = []
        
        for i in range(1, len(daily_data)):
            curr_date = daily_data.iloc[i]['quoteDate']
            prev_date = daily_data.iloc[i-1]['quoteDate']
            
            # Skip if dates are too far apart
            if (curr_date - prev_date).days > 7:
                continue
            
            # Calculate all possible ratios
            ratios = {
                'strike': daily_data.iloc[i-1]['strike'] / daily_data.iloc[i]['strike'],
                'price': daily_data.iloc[i-1]['stockClose'] / daily_data.iloc[i]['stockClose'],
                'volume_change': daily_data.iloc[i]['volume'] / daily_data.iloc[i-1]['volume'],
                'stockvolume_change': daily_data.iloc[i]['stockVolume'] / daily_data.iloc[i-1]['stockVolume'],
                'options_change': daily_data.iloc[i]['num_options'] / daily_data.iloc[i-1]['num_options']
            }
            
            # Clean ratios
            ratios = {k: v for k, v in ratios.items() if np.isfinite(v) and v > 0}
            
            # Check strike ratio first
            if 'strike' in ratios:
                split_ratio, split_type = self.identify_split_ratio(ratios['strike'])
                
                if split_ratio:
                    # Confirmatory checks
                    confirmations = []
                    
                    # Price alignment
                    if 'price' in ratios:
                        price_alignment = abs(1 - (ratios['price'] / split_ratio)) < 0.25
                        if price_alignment:
                            confirmations.append('price_aligned')
                    
                    # Volume spike
                    if ratios.get('volume_change', 1.0) > 1.2:
                        confirmations.append('volume_spike')
                    
                    # Stock volume spike
                    if ratios.get('stockvolume_change', 1.0) > 1.2:
                        confirmations.append('stock_volume_spike')
                    
                    # Options count change
                    if ratios.get('options_change', 1.0) > 1.2:
                        confirmations.append('options_increase')
                    
                    # If we have at least two confirmations
                    if len(confirmations) >= 2:
                        # Get surrounding data for context
                        window_start = max(0, i-2)
                        window_end = min(len(daily_data), i+3)
                        context_data = daily_data.iloc[window_start:window_end][
                            ['quoteDate', 'stockClose', 'stockAdjClose', 'strike']
                        ].to_dict('records')
                        
                        # Record the split
                        potential_splits.append({
                            'ticker': ticker,
                            'date': curr_date,
                            'split_ratio': split_ratio,
                            'split_type': split_type,
                            'confirmations': confirmations,
                            'confidence_score': len(confirmations),
                            'observed_ratios': ratios,
                            'context_data': context_data
                        })
        
        return potential_splits

    def detect_splits(self) -> pd.DataFrame:
        """Detect all potential splits with comprehensive analysis."""
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        all_splits = []
        verified_detections = set()
        
        # Process each ticker
        for ticker in tqdm(df['ticker'].unique(), desc="Analyzing tickers"):
            ticker_data = df[df['ticker'] == ticker].copy()
            potential_splits = self.analyze_ticker_data(ticker, ticker_data)
            
            # Check against verified splits
            if ticker in self.verified_splits:
                verified_split = self.verified_splits[ticker]
                verified_date = pd.to_datetime(verified_split['date'])
                
                detected = False
                for split in potential_splits:
                    if abs((split['date'] - verified_date).days) <= 7:
                        detected = True
                        verified_detections.add(ticker)
                        # Mark as verified in the data
                        split['verified'] = True
                        break
            
            all_splits.extend(potential_splits)
        
        if not all_splits:
            self.logger.warning("No splits detected!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        splits_df = pd.DataFrame(all_splits)
        
        # Convert complex columns to strings
        splits_df['observed_ratios'] = splits_df['observed_ratios'].apply(str)
        splits_df['context_data'] = splits_df['context_data'].apply(str)
        splits_df['confirmations'] = splits_df['confirmations'].apply(lambda x: ';'.join(x))
        
        # Sort by date within each ticker
        splits_df = splits_df.sort_values(['ticker', 'date'])
        
        # Save to CSV
        splits_df.to_csv(self.output_path, index=False)
        
        # Log summary
        self.logger.info(f"\nFound {len(splits_df)} potential splits across {splits_df['ticker'].nunique()} tickers")
        
        # Validation summary
        self.logger.info("\nValidation against known splits:")
        for ticker, split in self.verified_splits.items():
            status = "✓ Detected" if ticker in verified_detections else "✗ Not detected"
            self.logger.info(f"{ticker} ({split['date']}): {status}")
        
        # Additional splits found
        additional_splits = splits_df[
            ~splits_df['ticker'].isin(self.verified_splits.keys())
        ]
        if len(additional_splits) > 0:
            self.logger.info("\nAdditional potential splits found:")
            for _, split in additional_splits.iterrows():
                self.logger.info(
                    f"{split['ticker']}: {split['split_type']} on {split['date'].date()} "
                    f"(confidence: {split['confidence_score']})"
                )
        
        return splits_df

if __name__ == "__main__":
    detector = FinalSplitDetector('data_files/option_data_no_BACC.csv')
    final_splits = detector.detect_splits()