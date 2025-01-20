import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

class FinalSplitFixer:
    def __init__(self, data_path: str, output_path: str = None):
        self.data_path = data_path
        self.output_path = output_path or self._get_default_output_path(data_path)
        self.setup_logging()
        
        # Split definitions with exact specifications
        self.splits = {
            'TSLA': {
                'date': '2022-08-25',
                'ratio': 3.0,
                'type': 'forward',
                'fields_to_divide': [
                    'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                    'stockClose_ewm_5d', 'stockClose_ewm_15d',
                    'stockClose_ewm_45d', 'stockClose_ewm_135d',
                    'strike', 'lastPrice', 'strikeDelta'
                ]
            },
            'SHOP': {
                'date': '2022-07-04',
                'ratio': 10.0,
                'type': 'forward',
                'fields_to_divide': [
                    'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                    'stockClose_ewm_5d'
                ]
            },
            'CGC': {
                'date': '2023-12-20',
                'ratio': 10.0,  # Using 10.0 for 1:10 reverse split
                'type': 'reverse',
                'fields_to_divide': [
                    'strike', 'lastPrice', 'bid'
                ]
            }
        }

    def _get_default_output_path(self, input_path: str) -> str:
        path = Path(input_path)
        return str(path.parent / f"{path.stem}_split_adjusted_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def verify_price_continuity(self, ticker_data: pd.DataFrame, field: str, split_date: pd.Timestamp) -> dict:
        """Verify price continuity around split date."""
        window = 3  # days
        
        pre_split = ticker_data[
            (ticker_data['quoteDate'] < split_date) & 
            (ticker_data['quoteDate'] >= split_date - timedelta(days=window))
        ][field].median()
        
        post_split = ticker_data[
            (ticker_data['quoteDate'] > split_date) & 
            (ticker_data['quoteDate'] <= split_date + timedelta(days=window))
        ][field].median()
        
        if pre_split > 0 and post_split > 0:
            discontinuity = abs(1 - (post_split / pre_split))
            return {
                'pre_value': pre_split,
                'post_value': post_split,
                'discontinuity': discontinuity
            }
        return None

    def apply_split_adjustments(self) -> pd.DataFrame:
        """Apply split adjustments with thorough validation."""
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert date columns
        date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.split('+').str[0])
        
        df_adjusted = df.copy()
        
        for ticker, split_info in self.splits.items():
            split_date = pd.to_datetime(split_info['date'])
            split_ratio = split_info['ratio']
            split_type = split_info['type']
            
            self.logger.info(f"\nProcessing {ticker} {split_type} split on {split_date.date()}")
            
            # Get pre-split data
            ticker_mask = df_adjusted['ticker'] == ticker
            pre_split_mask = ticker_mask & (df_adjusted['quoteDate'] < split_date)
            affected_records = pre_split_mask.sum()
            
            if affected_records == 0:
                self.logger.warning(f"No pre-split records found for {ticker}")
                continue
            
            self.logger.info(f"Adjusting {affected_records:,} records")
            
            # Verify before adjustment
            for field in split_info['fields_to_divide']:
                if field not in df_adjusted.columns:
                    continue
                    
                before_metrics = self.verify_price_continuity(
                    df_adjusted[ticker_mask], field, split_date
                )
                
                if before_metrics:
                    self.logger.info(f"\n{field} before adjustment:")
                    self.logger.info(f"  Pre-split median:  {before_metrics['pre_value']:.2f}")
                    self.logger.info(f"  Post-split median: {before_metrics['post_value']:.2f}")
                    self.logger.info(f"  Discontinuity: {before_metrics['discontinuity']:.1%}")
                
                    # Apply adjustment
                    adjustment_factor = (1/split_ratio if split_type == 'forward' else split_ratio)
                    mask = pre_split_mask & df_adjusted[field].notna()
                    df_adjusted.loc[mask, field] *= adjustment_factor
                    
                    # Verify after adjustment
                    after_metrics = self.verify_price_continuity(
                        df_adjusted[ticker_mask], field, split_date
                    )
                    
                    if after_metrics:
                        self.logger.info(f"\n{field} after adjustment:")
                        self.logger.info(f"  Pre-split median:  {after_metrics['pre_value']:.2f}")
                        self.logger.info(f"  Post-split median: {after_metrics['post_value']:.2f}")
                        self.logger.info(f"  Discontinuity: {after_metrics['discontinuity']:.1%}")
                        
                        improvement = before_metrics['discontinuity'] - after_metrics['discontinuity']
                        self.logger.info(f"  Improvement: {improvement:.1%}")
        
        # Save adjusted data
        self.logger.info(f"\nSaving adjusted data to {self.output_path}")
        df_adjusted.to_csv(self.output_path, index=False)
        
        return df_adjusted

if __name__ == "__main__":
    fixer = FinalSplitFixer('data_files/option_data_no_BACC.csv')
    adjusted_data = fixer.apply_split_adjustments()