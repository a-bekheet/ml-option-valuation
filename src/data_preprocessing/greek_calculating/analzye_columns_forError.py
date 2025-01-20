import pandas as pd
import numpy as np

def analyze_random_rows():
    # Read the data
    df = pd.read_csv('/Users/bekheet/dev/option-ml-prediction/data_files/option_data.csv')
    
    # Select random rows (using different random seeds for diversity)
    samples = []
    for seed in [42, 123, 456, 789, 999]:
        np.random.seed(seed)
        samples.append(df.iloc[np.random.randint(len(df))])
    
    print("Detailed Analysis of Random Rows:")
    print("=" * 80)
    
    for i, row in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print("-" * 40)
        print(f"Contract Symbol: {row['contractSymbol']}")
        print(f"Option Type: {'Call' if 'C' in row['contractSymbol'] else 'Put'}")
        print(f"Strike: ${row['strike']:.2f}")
        print(f"Stock Price: ${row['stockClose']:.2f}")
        print(f"Strike Delta: {row['strikeDelta']:.6f}")
        print(f"Days to Expiry: {row['daysToExpiry']}")
        print(f"In The Money: {row['inTheMoney']}")
        print(f"Implied Volatility: {row['impliedVolatility']:.4f}")
        print(f"Bid/Ask: ${row['bid']:.2f}/${row['ask']:.2f}")
        
        # Add analysis notes
        print("\nAnalysis:")
        if abs(row['strikeDelta'] - row['stockClose']) < 1:
            print("WARNING: strikeDelta is very close to stockClose!")
        if row['inTheMoney'] and 'C' in row['contractSymbol']:
            print("Call is ITM - Delta should be > 0.5")
        if row['inTheMoney'] and 'P' in row['contractSymbol']:
            print("Put is ITM - Delta should be < -0.5")

    # Additional overall statistics
    print("\nOverall Column Statistics:")
    print("=" * 80)
    for col in ['strikeDelta', 'stockClose', 'strike']:
        print(f"\n{col} stats:")
        print(df[col].describe())

# Run the analysis
analyze_random_rows()