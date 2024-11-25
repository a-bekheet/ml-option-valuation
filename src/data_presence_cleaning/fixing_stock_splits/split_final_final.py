import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple

class FinalSplitAdjuster:
    def __init__(self, input_file: str, output_file: str = None):
        """Initialize with verified split information."""
        self.input_file = input_file
        self.output_file = output_file or self._get_default_output_path(input_file)
        self.setup_logging()
        
        # Verified splits with exact specifications
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
        
        # Fields that should never be adjusted
        self.no_adjustment_fields = {
            'contractSymbol', 'lastTradeDate', 'quoteDate', 'expiryDate',
            'percentChange', 'impliedVolatility', 'inTheMoney',
            'contractSize', 'currency', 'daysToExpiry',
            'stockAdjClose'  # Already split-adjusted
        }

    def _get_default_output_path(self, input_path: str) -> str:
        """Generate output path with timestamp."""
        path = Path(input_path)
        return str(path.parent / f"{path.stem}_final_adjusted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    def setup_logging(self):
        """Configure logging with detailed format."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def verify_adjustment_needed(
        self,
        data: pd.DataFrame,
        field: str,
        split_date: pd.Timestamp,
        window_days: int = 3
    ) -> Tuple[bool, Dict]:
        """
        Verify if adjustment is needed by checking discontinuity.
        Returns (needs_adjustment, metrics).
        """
        pre_split = data[
            (data['quoteDate'] < split_date) &
            (data['quoteDate'] >= split_date - timedelta(days=window_days))
        ][field].median()
        
        post_split = data[
            (data['quoteDate'] > split_date) &
            (data['quoteDate'] <= split_date + timedelta(days=window_days))
        ][field].median()
        
        if pre_split > 0 and post_split > 0:
            discontinuity = abs(1 - (post_split / pre_split))
            return discontinuity > 0.1, {
                'pre_value': pre_split,
                'post_value': post_split,
                'discontinuity': discontinuity
            }
        return False, {}

    def adjust_splits(self) -> pd.DataFrame:
        """
        Apply split adjustments with validation and logging.
        """
        self.logger.info(f"Loading data from {self.input_file}")
        df = pd.read_csv(self.input_file)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        df_adjusted = df.copy()
        adjustments_made = []
        
        # Process each split
        for ticker, split_info in self.splits.items():
            split_date = pd.to_datetime(split_info['date'])
            split_ratio = split_info['ratio']
            split_type = split_info['type']
            
            self.logger.info(f"\nProcessing {ticker} {split_type} split on {split_date.date()}")
            self.logger.info(f"Split ratio: {split_ratio if split_type == 'forward' else f'1:{1/split_ratio:.0f}'}")
            
            # Get data for this ticker
            ticker_mask = df_adjusted['ticker'] == ticker
            pre_split_mask = ticker_mask & (df_adjusted['quoteDate'] < split_date)
            affected_records = pre_split_mask.sum()
            
            if affected_records == 0:
                self.logger.warning(f"No pre-split records found for {ticker}")
                continue
            
            self.logger.info(f"Found {affected_records:,} pre-split records")
            
            # Process fields that need division
            for field in split_info['adjustments']['divide']:
                if field not in df_adjusted.columns:
                    continue
                    
                needs_adjustment, metrics = self.verify_adjustment_needed(
                    df_adjusted[ticker_mask], field, split_date
                )
                
                if needs_adjustment:
                    # Apply adjustment
                    mask = pre_split_mask & df_adjusted[field].notna()
                    original_values = df_adjusted.loc[mask, field].copy()
                    df_adjusted.loc[mask, field] /= split_ratio
                    
                    # Verify adjustment
                    _, after_metrics = self.verify_adjustment_needed(
                        df_adjusted[ticker_mask], field, split_date
                    )
                    
                    adjustments_made.append({
                        'ticker': ticker,
                        'field': field,
                        'operation': 'divide',
                        'ratio': split_ratio,
                        'before_discontinuity': metrics['discontinuity'],
                        'after_discontinuity': after_metrics.get('discontinuity', np.nan)
                    })
                    
                    self.logger.info(f"\nAdjusted {field}:")
                    self.logger.info(f"  Pre-adjustment:  {metrics['pre_value']:.2f}")
                    self.logger.info(f"  Post-adjustment: {metrics['post_value']:.2f}")
                    self.logger.info(f"  Sample values before: {original_values.iloc[:3].round(2).tolist()}")
                    self.logger.info(f"  Sample values after:  {df_adjusted.loc[mask, field].iloc[:3].round(2).tolist()}")
            
            # Process fields that need multiplication
            for field in split_info['adjustments']['multiply']:
                if field not in df_adjusted.columns:
                    continue
                    
                needs_adjustment, metrics = self.verify_adjustment_needed(
                    df_adjusted[ticker_mask], field, split_date
                )
                
                if needs_adjustment:
                    # Apply adjustment
                    mask = pre_split_mask & df_adjusted[field].notna()
                    original_values = df_adjusted.loc[mask, field].copy()
                    df_adjusted.loc[mask, field] *= split_ratio
                    
                    # Verify adjustment
                    _, after_metrics = self.verify_adjustment_needed(
                        df_adjusted[ticker_mask], field, split_date
                    )
                    
                    adjustments_made.append({
                        'ticker': ticker,
                        'field': field,
                        'operation': 'multiply',
                        'ratio': split_ratio,
                        'before_discontinuity': metrics['discontinuity'],
                        'after_discontinuity': after_metrics.get('discontinuity', np.nan)
                    })
                    
                    self.logger.info(f"\nAdjusted {field}:")
                    self.logger.info(f"  Pre-adjustment:  {metrics['pre_value']:.2f}")
                    self.logger.info(f"  Post-adjustment: {metrics['post_value']:.2f}")
                    self.logger.info(f"  Sample values before: {original_values.iloc[:3].round(2).tolist()}")
                    self.logger.info(f"  Sample values after:  {df_adjusted.loc[mask, field].iloc[:3].round(2).tolist()}")
        
        # Save adjusted data
        self.logger.info(f"\nSaving adjusted data to {self.output_file}")
        df_adjusted.to_csv(self.output_file, index=False)
        
        # Log adjustment summary
        if adjustments_made:
            self.logger.info("\nAdjustment Summary:")
            for adj in adjustments_made:
                self.logger.info(
                    f"{adj['ticker']} - {adj['field']}: "
                    f"{adj['operation']} by {adj['ratio']}, "
                    f"Discontinuity improved from {adj['before_discontinuity']:.1%} "
                    f"to {adj['after_discontinuity']:.1%}"
                )
        else:
            self.logger.info("\nNo adjustments were necessary")
        
        return df_adjusted

if __name__ == "__main__":
    adjuster = FinalSplitAdjuster('data_files/option_data_no_BACC.csv')
    adjusted_data = adjuster.adjust_splits()