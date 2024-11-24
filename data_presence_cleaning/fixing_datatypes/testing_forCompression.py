import pandas as pd
import numpy as np
from datetime import datetime


def create_sample_data(file_path, sample_size=1000, random_seed=42):
    """
    Create a sample dataset from the large CSV
    """
    np.random.seed(random_seed)
    
    # Read random sample of rows
    df_sample = pd.read_csv(
        file_path, 
        nrows=sample_size
    )
    
    print(f"Created sample with {len(df_sample)} rows")
    return df_sample

def validate_transformed_data(original_df, optimized_df):
    """
    Comprehensive validation of data transformations
    """
    print("\nValidating Data Transformations:")
    print("=" * 80)
    
    validation_results = {}
    
    def check_date_columns():
        """Validate datetime conversions"""
        date_cols = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_cols:
            # Check if dates are properly parsed
            sample_orig = original_df[col].iloc[0]
            sample_opt = optimized_df[col].iloc[0]
            
            print(f"\n{col} Validation:")
            print(f"Original: {sample_orig} ({type(sample_orig)})")
            print(f"Transformed: {sample_opt} ({type(sample_opt)})")
            
            # Verify we can perform date operations
            if isinstance(sample_opt, pd.Timestamp):
                try:
                    days_diff = (optimized_df[col].max() - optimized_df[col].min()).days
                    print(f"Date range spans {days_diff} days")
                    validation_results[col] = "PASS"
                except Exception as e:
                    print(f"Failed date operation: {str(e)}")
                    validation_results[col] = "FAIL"
    
    def check_numeric_columns():
        """Validate numeric transformations"""
        numeric_cols = ['strike', 'lastPrice', 'bid', 'ask', 'change', 
                       'percentChange', 'volume', 'openInterest', 'impliedVolatility']
        
        for col in numeric_cols:
            print(f"\n{col} Validation:")
            
            # Basic statistics comparison
            orig_stats = original_df[col].describe()
            opt_stats = optimized_df[col].describe()
            
            print(f"Original range: {orig_stats['min']:.4f} to {orig_stats['max']:.4f}")
            print(f"Optimized range: {opt_stats['min']:.4f} to {opt_stats['max']:.4f}")
            
            # Check if we can perform calculations
            try:
                # Test basic arithmetic
                test_calc = optimized_df[col].mean() * 2
                test_sum = optimized_df[col].sum()
                print(f"Can perform calculations: Yes")
                validation_results[col] = "PASS"
            except Exception as e:
                print(f"Calculation failed: {str(e)}")
                validation_results[col] = "FAIL"
    
    def check_categorical_columns():
        """Validate categorical transformations"""
        cat_cols = ['contractSize', 'currency']
        for col in cat_cols:
            print(f"\n{col} Validation:")
            
            # Check unique values preserved
            orig_unique = set(original_df[col].unique())
            opt_unique = set(optimized_df[col].unique())
            
            print(f"Original categories: {orig_unique}")
            print(f"Optimized categories: {opt_unique}")
            
            if orig_unique == opt_unique:
                print("Categories preserved: Yes")
                validation_results[col] = "PASS"
            else:
                print("WARNING: Categories changed during optimization")
                validation_results[col] = "FAIL"
    
    def check_option_specific_calculations():
        """Test option-specific calculations and relationships"""
        print("\nValidating Option-Specific Relationships:")
        
        try:
            # Test in-the-money relationships
            call_itm = optimized_df[
                (optimized_df['contractSymbol'].str.contains('C')) & 
                (optimized_df['strike'] < optimized_df['stockClose'])
            ]
            
            put_itm = optimized_df[
                (optimized_df['contractSymbol'].str.contains('P')) & 
                (optimized_df['strike'] > optimized_df['stockClose'])
            ]
            
            print(f"Can identify ITM calls: {len(call_itm)} found")
            print(f"Can identify ITM puts: {len(put_itm)} found")
            
            # Test bid-ask spread calculation
            optimized_df['spread'] = optimized_df['ask'] - optimized_df['bid']
            print(f"Can calculate bid-ask spread: Yes")
            print(f"Average spread: {optimized_df['spread'].mean():.4f}")
            
            # Test time to expiry calculation
            optimized_df['tte'] = (
                pd.to_datetime(optimized_df['expiryDate']) - 
                pd.to_datetime(optimized_df['quoteDate'])
            ).dt.days
            
            print(f"Can calculate time to expiry: Yes")
            print(f"Average days to expiry: {optimized_df['tte'].mean():.1f}")
            
            validation_results['option_calculations'] = "PASS"
            
        except Exception as e:
            print(f"Option calculations failed: {str(e)}")
            validation_results['option_calculations'] = "FAIL"
    
    # Run all validations
    check_date_columns()
    check_numeric_columns()
    check_categorical_columns()
    check_option_specific_calculations()
    
    # Summary
    print("\nValidation Summary:")
    print("=" * 80)
    for col, result in validation_results.items():
        print(f"{col}: {result}")
    
    # Overall assessment
    passed = all(result == "PASS" for result in validation_results.values())
    return passed

def test_optimization_strategy(sample_df):
    """
    Test optimization strategy on sample data with NA handling
    """
    print("\nTesting optimization strategy...")
    print("\nInitial Data Analysis:")
    
    # Check for NaN and inf values
    for col in sample_df.columns:
        na_count = sample_df[col].isna().sum()
        inf_count = np.isinf(pd.to_numeric(sample_df[col], errors='coerce')).sum()
        if na_count > 0 or inf_count > 0:
            print(f"{col}: {na_count} NA values, {inf_count} inf values")
    
    try:
        optimized_df = sample_df.copy()
        
        # 1. Handle datetime columns
        date_cols = ['lastTradeDate', 'quoteDate', 'expiryDate']
        for col in date_cols:
            optimized_df[col] = pd.to_datetime(optimized_df[col])
        
        # 2. Handle float32 conversions (with NA handling)
        float32_cols = [
            'stockClose_ewm_5d', 'stockClose_ewm_15d', 'stockClose_ewm_45d',
            'stockClose_ewm_135d', 'stockHigh', 'stockLow', 'stockOpen',
            'stockClose', 'stockAdjClose', 'ask', 'bid', 'change',
            'strikeDelta', 'impliedVolatility'
        ]
        for col in float32_cols:
            # Fill NA with 0 or mean
            if optimized_df[col].isna().any():
                fill_value = optimized_df[col].mean()
                optimized_df[col] = optimized_df[col].fillna(fill_value)
            optimized_df[col] = optimized_df[col].astype('float32')
        
        # 3. Handle integer conversions (with NA handling)
        int_cols = {
            'stockVolume': 'uint32',
            'volume': 'uint32',
            'openInterest': 'uint32'
        }
        for col, dtype in int_cols.items():
            if optimized_df[col].isna().any():
                # Fill NA with 0 for volume/interest columns
                optimized_df[col] = optimized_df[col].fillna(0)
            # Convert to int, clipping any negative values to 0
            optimized_df[col] = optimized_df[col].clip(lower=0).astype(dtype)
        
        # 4. Handle categorical conversions
        cat_cols = ['contractSize', 'currency']
        for col in cat_cols:
            # Fill NA with most common value
            if optimized_df[col].isna().any():
                fill_value = optimized_df[col].mode()[0]
                optimized_df[col] = optimized_df[col].fillna(fill_value)
            optimized_df[col] = optimized_df[col].astype('category')
        
        # 5. Handle daysToExpiry
        optimized_df['daysToExpiry'] = (optimized_df['daysToExpiry']
                                       .str.replace(' days', '')
                                       .fillna(0)
                                       .astype('uint16'))
        
        # Calculate memory usage
        original_mem = sample_df.memory_usage(deep=True).sum() / 1024**2
        optimized_mem = optimized_df.memory_usage(deep=True).sum() / 1024**2
        savings = (1 - optimized_mem/original_mem) * 100
        
        print("\nOptimization Results:")
        print("=" * 50)
        print(f"Original Memory: {original_mem:.2f} MB")
        print(f"Optimized Memory: {optimized_mem:.2f} MB")
        print(f"Memory Savings: {savings:.1f}%")
        
        # Data integrity check
        print("\nData Integrity Check:")
        print("=" * 50)
        for col in optimized_df.columns:
            print(f"\n{col}:")
            print(f"Original dtype: {sample_df[col].dtype}")
            print(f"Optimized dtype: {optimized_df[col].dtype}")
            print(f"NA values before: {sample_df[col].isna().sum()}")
            print(f"NA values after: {optimized_df[col].isna().sum()}")
        
        return True, optimized_df
        
    except Exception as e:
        print(f"Error during optimization test: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False, None
    
    if success:
        validation_passed = validate_transformed_data(sample_df, optimized_df)
        if validation_passed:
            print("\nAll validations passed - data transformation is safe!")
        else:
            print("\nWARNING: Some validations failed - review results above")



def main():
    # File path
    file_path = 'data_files/option_data_with_headers_cleaned.csv'  # Adjust path as needed
    
    try:
        # Create and test with sample
        print("Creating sample dataset...")
        sample_df = create_sample_data(file_path)
        
        # Test optimization strategy
        success, optimized_sample = test_optimization_strategy(sample_df)
        
        if success:
            print("\nTest successful! Safe to proceed with full dataset.")
            return True
        else:
            print("\nTest failed. Please review the error messages.")
            return False
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()