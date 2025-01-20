import pandas as pd
import numpy as np

def analyze_and_clean_delta():
    df = pd.read_csv('data_files/option_data_with_headers_cleaned.csv')
    
    print("\nBefore Cleaning:")
    print("Overall Delta Stats:")
    print(df['strikeDelta'].describe())
    
    # Identify suspicious values
    suspicious = df[abs(df['strikeDelta']) > 1]
    print(f"\nNumber of suspicious delta values: {len(suspicious)}")
    print("\nSample of suspicious entries:")
    print(suspicious[['contractSymbol', 'strikeDelta', 'strike', 'stockClose', 'inTheMoney']].head())
    
    # Clean delta values
    df['delta_cleaned'] = df['strikeDelta'].clip(-1, 1)  # Clip to valid range
    
    # Split into calls and puts
    calls = df[df['contractSymbol'].str.contains('C')]
    puts = df[df['contractSymbol'].str.contains('P')]
    
    print("\nAfter Cleaning:")
    print("\nCall Options Delta Analysis:")
    print(f"Delta Range: {calls['delta_cleaned'].min():.3f} to {calls['delta_cleaned'].max():.3f}")
    print(f"Average Delta for ITM calls: {calls[calls['inTheMoney']]['delta_cleaned'].mean():.3f}")
    print(f"Average Delta for OTM calls: {calls[~calls['inTheMoney']]['delta_cleaned'].mean():.3f}")
    
    print("\nPut Options Delta Analysis:")
    print(f"Delta Range: {puts['delta_cleaned'].min():.3f} to {puts['delta_cleaned'].max():.3f}")
    print(f"Average Delta for ITM puts: {puts[puts['inTheMoney']]['delta_cleaned'].mean():.3f}")
    print(f"Average Delta for OTM puts: {puts[~puts['inTheMoney']]['delta_cleaned'].mean():.3f}")
    
    # Calculate delta using Black-Scholes for verification
    # Note: This requires additional data like risk-free rate
    
    # Save cleaned data
    df.to_csv('option_data_cleaned_delta.csv', index=False)
    
    # Return problematic rows for investigation
    return suspicious

# Run analysis
suspicious_data = analyze_and_clean_delta()