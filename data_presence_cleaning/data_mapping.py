import pandas as pd
import numpy as np

def analyze_option_data_columns(file_path='/Users/bekheet/dev/option-ml-prediction/data_files/option_data_with_headers.csv'):
    """
    Analyze column presence in the options data
    """
    # Define expected column names
    expected_columns = [
        'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 
        'ask', 'change', 'percentChange', 'volume', 'openInterest',
        'impliedVolatility', 'inTheMoney', 'contractSize', 'currency',
        'quoteDate', 'expiryDate', 'daysToExpiry', 'stockVolume',
        'stockClose', 'stockAdjClose', 'stockOpen', 'stockHigh',
        'stockLow', 'strikeDelta', 'stockClose_ewm_5d', 'stockClose_ewm_15d',
        'stockClose_ewm_45d', 'stockClose_ewm_135d'
    ]
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Create presence analysis DataFrame
    presence_data = pd.DataFrame({
        'Column Name': expected_columns,
        'Present': [df[col].notna().sum() for col in df.columns],
        'Missing': [df[col].isna().sum() for col in df.columns],
    })
    
    # Calculate percentage present
    presence_data['Percentage Present'] = (
        presence_data['Present'] / (presence_data['Present'] + presence_data['Missing']) * 100
    )
    
    # Add data types
    presence_data['Data Type'] = [str(df[col].dtype) for col in df.columns]
    
    # Add recommendations
    presence_data['Recommendation'] = presence_data['Percentage Present'].apply(
        lambda x: 'Keep' if x > 99.5 
        else 'Review' if x > 98 
        else 'Consider Dropping'
    )
    
    # Print summary statistics
    print("\nColumn Analysis Summary:")
    print("-" * 80)
    print(presence_data.to_string(index=False))
    
    # Print data quality concerns
    print("\nData Quality Concerns:")
    print("-" * 80)
    quality_issues = presence_data[presence_data['Percentage Present'] < 99.5]
    for _, row in quality_issues.iterrows():
        print(f"Column '{row['Column Name']}' has {row['Missing']} missing values "
              f"({100 - row['Percentage Present']:.2f}% missing)")
    
    # Save the analysis
    presence_data.to_csv('column_analysis_recommendations.csv', index=False)
    
    return presence_data

# Run the analysis
analysis = analyze_option_data_columns()