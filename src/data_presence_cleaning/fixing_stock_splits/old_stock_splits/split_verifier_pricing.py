import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

class SplitAdjustmentVerifier:
    def __init__(self, original_path: str, adjusted_path: str):
        self.original_path = original_path
        self.adjusted_path = adjusted_path
        self.setup_logging()
        
        # Known splits for verification
        self.verified_splits = {
            'TSLA': {
                'date': '2022-08-25',
                'ratio': 3.0,
                'fields': [
                    'stockClose', 'stockOpen', 'stockHigh', 'stockLow',
                    'strike', 'lastPrice', 'strikeDelta'
                ]
            },
            'SHOP': {
                'date': '2022-07-04',
                'ratio': 10.0,
                'fields': [
                    'stockClose', 'stockOpen', 'stockHigh', 'stockLow'
                ]
            },
            'CGC': {
                'date': '2023-12-20',
                'ratio': 0.1,  # 1:10 reverse
                'fields': [
                    'strike', 'lastPrice', 'bid'
                ]
            }
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both original and adjusted datasets."""
        self.logger.info("Loading datasets...")
        
        # Load data
        df_orig = pd.read_csv(self.original_path)
        df_adj = pd.read_csv(self.adjusted_path)
        
        # Convert date columns
        date_cols = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_cols:
            if col in df_orig.columns:
                df_orig[col] = pd.to_datetime(df_orig[col].str.split('+').str[0])
                df_adj[col] = pd.to_datetime(df_adj[col].str.split('+').str[0])
        
        return df_orig, df_adj

    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            return hist
        except Exception as e:
            self.logger.warning(f"Error fetching {ticker} historical data: {str(e)}")
            return pd.DataFrame()

    def verify_adjustment_ratios(
        self,
        df_orig: pd.DataFrame,
        df_adj: pd.DataFrame,
        ticker: str,
        split_info: dict,
        window_days: int = 5
    ) -> Dict:
        """Verify adjustment ratios for a specific ticker."""
        split_date = pd.to_datetime(split_info['date'])
        expected_ratio = split_info['ratio']
        
        results = {
            'ticker': ticker,
            'split_date': split_date,
            'fields': {}
        }
        
        for field in split_info['fields']:
            if field not in df_orig.columns or field not in df_adj.columns:
                continue
            
            # Get data around split
            orig_pre = df_orig[
                (df_orig['ticker'] == ticker) & 
                (df_orig['quoteDate'] < split_date) & 
                (df_orig['quoteDate'] >= split_date - timedelta(days=window_days))
            ][field].median()
            
            orig_post = df_orig[
                (df_orig['ticker'] == ticker) & 
                (df_orig['quoteDate'] > split_date) & 
                (df_orig['quoteDate'] <= split_date + timedelta(days=window_days))
            ][field].median()
            
            adj_pre = df_adj[
                (df_adj['ticker'] == ticker) & 
                (df_adj['quoteDate'] < split_date) & 
                (df_adj['quoteDate'] >= split_date - timedelta(days=window_days))
            ][field].median()
            
            adj_post = df_adj[
                (df_adj['ticker'] == ticker) & 
                (df_adj['quoteDate'] > split_date) & 
                (df_adj['quoteDate'] <= split_date + timedelta(days=window_days))
            ][field].median()
            
            if all(v > 0 for v in [orig_pre, orig_post, adj_pre, adj_post]):
                orig_ratio = orig_post / orig_pre
                adj_ratio = adj_post / adj_pre
                
                # For reverse splits, invert the expected ratio
                compare_ratio = 1/expected_ratio if expected_ratio < 1 else expected_ratio
                
                # Calculate errors
                orig_error = abs(1 - (orig_ratio * compare_ratio))
                adj_error = abs(1 - adj_ratio)
                
                results['fields'][field] = {
                    'original_ratio': orig_ratio,
                    'adjusted_ratio': adj_ratio,
                    'original_error': orig_error,
                    'adjusted_error': adj_error,
                    'improvement': orig_error - adj_error
                }
        
        return results

    def compare_to_historical(
        self,
        df_adj: pd.DataFrame,
        ticker: str,
        split_info: dict,
        window_days: int = 10
    ) -> Dict:
        """Compare adjusted prices to historical data."""
        split_date = pd.to_datetime(split_info['date'])
        start_date = split_date - timedelta(days=window_days)
        end_date = split_date + timedelta(days=window_days)
        
        # Get historical data
        hist_data = self.get_historical_data(ticker, start_date, end_date)
        if hist_data.empty:
            return {}
        
        # Get adjusted data
        adj_data = df_adj[
            (df_adj['ticker'] == ticker) & 
            (df_adj['quoteDate'] >= start_date) & 
            (df_adj['quoteDate'] <= end_date)
        ].copy()
        
        results = {
            'ticker': ticker,
            'split_date': split_date,
            'price_comparison': {}
        }
        
        # Compare closing prices
        if 'stockClose' in adj_data.columns:
            adj_daily = adj_data.groupby('quoteDate')['stockClose'].mean()
            
            # Calculate correlation and error metrics
            common_dates = hist_data.index.intersection(adj_daily.index)
            if len(common_dates) > 0:
                hist_prices = hist_data.loc[common_dates, 'Close']
                adj_prices = adj_daily.loc[common_dates]
                
                correlation = np.corrcoef(hist_prices, adj_prices)[0, 1]
                mape = np.mean(np.abs((hist_prices - adj_prices) / hist_prices)) * 100
                
                results['price_comparison'] = {
                    'correlation': correlation,
                    'mape': mape,
                    'sample_comparison': {
                        str(date.date()): {
                            'historical': round(hist_prices[date], 2),
                            'adjusted': round(adj_prices[date], 2)
                        }
                        for date in common_dates[:5]  # Show first 5 days
                    }
                }
        
        return results

    def verify_adjustments(self) -> None:
        """Run comprehensive verification of split adjustments."""
        df_orig, df_adj = self.load_data()
        
        self.logger.info("\nVerifying split adjustments...")
        
        all_results = []
        for ticker, split_info in self.verified_splits.items():
            self.logger.info(f"\nAnalyzing {ticker}...")
            
            # Verify adjustment ratios
            ratio_results = self.verify_adjustment_ratios(df_orig, df_adj, ticker, split_info)
            
            # Compare to historical data
            hist_results = self.compare_to_historical(df_adj, ticker, split_info)
            
            # Log results
            self.logger.info(f"\n{ticker} Split Verification:")
            self.logger.info(f"Split Date: {split_info['date']}")
            self.logger.info(f"Split Ratio: {split_info['ratio']}")
            
            if ratio_results['fields']:
                self.logger.info("\nField Adjustments:")
                for field, metrics in ratio_results['fields'].items():
                    self.logger.info(f"\n{field}:")
                    self.logger.info(f"  Original discontinuity: {metrics['original_error']:.1%}")
                    self.logger.info(f"  Adjusted discontinuity: {metrics['adjusted_error']:.1%}")
                    self.logger.info(f"  Improvement: {metrics['improvement']:.1%}")
            
            if hist_results.get('price_comparison'):
                self.logger.info("\nHistorical Comparison:")
                self.logger.info(f"  Correlation: {hist_results['price_comparison']['correlation']:.3f}")
                self.logger.info(f"  Mean Absolute % Error: {hist_results['price_comparison']['mape']:.1f}%")
                self.logger.info("\nSample Price Comparison:")
                for date, prices in hist_results['price_comparison']['sample_comparison'].items():
                    self.logger.info(
                        f"  {date}: Historical ${prices['historical']}, "
                        f"Adjusted ${prices['adjusted']}"
                    )
            
            all_results.append({
                'ratio_results': ratio_results,
                'hist_results': hist_results
            })
        
        return all_results

if __name__ == "__main__":
    verifier = SplitAdjustmentVerifier(
        'data_files/option_data_no_BACC.csv',
        'data_files/option_data_no_BACC_split_adjusted_20241124_181851.csv'
    )
    verification_results = verifier.verify_adjustments()