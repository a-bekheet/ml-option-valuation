import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Set
import logging
from pathlib import Path

class PreciseStockSplitAdjuster:
    def __init__(self, data_path: str, output_path: str = None):
        self.data_path = data_path
        self.output_path = output_path or self._get_default_output_path(data_path)
        self.setup_logging()
        
        # Split definitions with exact adjustments needed
        self.split_adjustments = {
            'TSLA': {
                'date': '2022-08-25',
                'ratio': 3.0,
                'fields_to_adjust': {
                    'price_fields': [
                        'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                        'stockClose_ewm_5d', 'stockClose_ewm_15d',
                        'stockClose_ewm_45d', 'stockClose_ewm_135d',
                        'strike', 'lastPrice', 'strikeDelta'
                    ],
                    'volume_fields': []  # Already adjusted
                }
            },
            'SHOP': {
                'date': '2022-07-04',
                'ratio': 10.0,
                'fields_to_adjust': {
                    'price_fields': [
                        'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                        'stockClose_ewm_5d'
                    ],
                    'volume_fields': []  # Already adjusted
                }
            },
            'CGC': {
                'date': '2023-12-20',
                'ratio': 0.1,  # 1:10 reverse split
                'fields_to_adjust': {
                    'price_fields': [
                        'strike', 'lastPrice', 'bid'
                    ],
                    'volume_fields': []  # Already adjusted
                }
            }
            # AMZN and GOOGL excluded as they're already fully adjusted
        }
        
        # Fields that should never be adjusted
        self.no_adjust_fields = {
            'stockAdjClose',  # Already split-adjusted
            'contractSymbol', 'ticker', 'quoteDate', 'lastTradeDate', 'expiryDate',
            'daysToExpiry', 'contractSize', 'currency', 'percentChange',
            'impliedVolatility', 'inTheMoney'
        }

    def _get_default_output_path(self, input_path: str) -> str:
        """Generate default output path with precise naming."""
        path = Path(input_path)
        return str(path.parent / f"{path.stem}_split_adjusted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    def setup_logging(self):
        """Setup detailed logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_adjustment_needed(
        self,
        df: pd.DataFrame,
        ticker: str,
        field: str,
        split_date: pd.Timestamp,
        split_ratio: float,
        window_days: int = 3
    ) -> bool:
        """
        Validate if a field actually needs adjustment by comparing ratios.
        Returns True if adjustment is needed.
        """
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Get data around split date
        pre_split = ticker_data[
            (ticker_data['quoteDate'] < split_date) & 
            (ticker_data['quoteDate'] >= split_date - pd.Timedelta(days=window_days))
        ]
        
        post_split = ticker_data[
            (ticker_data['quoteDate'] > split_date) & 
            (ticker_data['quoteDate'] <= split_date + pd.Timedelta(days=window_days))
        ]
        
        if len(pre_split) == 0 or len(post_split) == 0:
            return False
            
        # Calculate median values
        pre_value = pre_split[field].median()
        post_value = post_split[field].median()
        
        if pre_value == 0 or post_value == 0 or pd.isna(pre_value) or pd.isna(post_value):
            return False
            
        # Calculate observed ratio
        observed_ratio = post_value / pre_value
        
        # For reverse splits, invert the expected ratio
        expected_ratio = split_ratio if split_ratio > 1 else (1 / split_ratio)
        
        # Check if the field needs adjustment (>10% difference from expected)
        ratio_diff = abs(1 - (observed_ratio * expected_ratio))
        needs_adjustment = ratio_diff > 0.1
        
        if needs_adjustment:
            self.logger.info(
                f"{ticker} {field}: Adjustment needed "
                f"(observed ratio: {observed_ratio:.2f}, expected: {1/expected_ratio:.2f})"
            )
        
        return needs_adjustment

    def adjust_splits(self) -> pd.DataFrame:
        """
        Perform precise split adjustments with validation.
        """
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        # Create copy for adjustments
        df_adjusted = df.copy()
        
        # Process each split
        for ticker, split_info in self.split_adjustments.items():
            split_date = pd.to_datetime(split_info['date'])
            split_ratio = split_info['ratio']
            
            self.logger.info(f"\nProcessing {ticker} "
                          f"{'split' if split_ratio > 1 else 'reverse split'} "
                          f"on {split_date.date()}")
            
            # Get pre-split data mask
            ticker_mask = df_adjusted['ticker'] == ticker
            pre_split_mask = ticker_mask & (df_adjusted['quoteDate'] < split_date)
            affected_records = pre_split_mask.sum()
            
            if affected_records == 0:
                self.logger.warning(f"No pre-split records found for {ticker}")
                continue
                
            self.logger.info(f"Found {affected_records:,} pre-split records")
            
            # Validate and adjust price fields
            for field in split_info['fields_to_adjust']['price_fields']:
                if field not in df_adjusted.columns:
                    continue
                    
                if self.validate_adjustment_needed(df_adjusted, ticker, field, 
                                                split_date, split_ratio):
                    # For regular splits: divide by ratio
                    # For reverse splits: multiply by inverse ratio
                    adjustment_factor = (1 / split_ratio) if split_ratio > 1 else split_ratio
                    
                    # Only adjust non-null values
                    mask = pre_split_mask & df_adjusted[field].notna()
                    original_values = df_adjusted.loc[mask, field].copy()
                    df_adjusted.loc[mask, field] *= adjustment_factor
                    
                    # Verify adjustment
                    adjusted_values = df_adjusted.loc[mask, field]
                    self.logger.info(
                        f"Adjusted {field}: "
                        f"Sample before [{original_values.iloc[:3].round(2).tolist()}], "
                        f"after [{adjusted_values.iloc[:3].round(2).tolist()}]"
                    )
            
            # Verify against stockAdjClose
            if 'stockClose' in split_info['fields_to_adjust']['price_fields']:
                adj_close_ratio = (
                    df_adjusted[df_adjusted['quoteDate'] > split_date]['stockAdjClose'].mean() /
                    df_adjusted[df_adjusted['quoteDate'] < split_date]['stockAdjClose'].mean()
                )
                self.logger.info(f"AdjClose ratio verification: {adj_close_ratio:.2f}")
        
        # Save adjusted data
        self.logger.info(f"\nSaving adjusted data to {self.output_path}")
        df_adjusted.to_csv(self.output_path, index=False)
        
        # Log adjustment summary
        self.logger.info("\nAdjustment Summary:")
        for ticker, split_info in self.split_adjustments.items():
            self.logger.info(f"\n{ticker}:")
            self.logger.info(f"  Split Date: {split_info['date']}")
            ratio_str = split_info['ratio'] if split_info['ratio'] > 1 else f'1:{1/split_info["ratio"]:.0f}'
            self.logger.info(f"  Ratio: {ratio_str}")
            self.logger.info("  Fields adjusted:")
            for field in split_info['fields_to_adjust']['price_fields']:
                if field in df_adjusted.columns:
                    self.logger.info(f"    - {field}")
        
        return df_adjusted

if __name__ == "__main__":
    adjuster = PreciseStockSplitAdjuster('data_files/option_data_no_BACC.csv')
    adjusted_data = adjuster.adjust_splits()