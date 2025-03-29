# src/data_preprocessing/greeks_calculator.py

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# N(x) is the cumulative distribution function for the standard normal distribution
N = norm.cdf
# N'(x) is the probability density function for the standard normal distribution
N_prime = norm.pdf

def calculate_d1_d2(S, K, T, r, sigma):
    """
    Calculates d1 and d2 parameters used in Black-Scholes.

    Args:
        S (pd.Series): Underlying asset price.
        K (pd.Series): Strike price.
        T (pd.Series): Time to expiry in years.
        r (pd.Series): Risk-free interest rate (annualized decimal).
        sigma (pd.Series): Volatility (annualized decimal).

    Returns:
        tuple: (d1, d2) as pandas Series.
    """
    # Ensure T and sigma are not zero to avoid division errors
    # Replace T=0 with a very small number (epsilon)
    epsilon = 1e-9
    T_safe = np.maximum(T, epsilon)
    sigma_safe = np.maximum(sigma, epsilon)

    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    return d1, d2

def calculate_greeks_bs(df: pd.DataFrame, option_type_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculates Option Greeks using Black-Scholes formulas as an approximation.
    Assumes necessary columns (stockClose, strike, daysToExpiry, risk_free_rate,
    impliedVolatility) exist in the DataFrame.

    IMPORTANT: These are Black-Scholes Greeks, which are technically for European options.
               They serve as an approximation for American options but do not account
               for the early exercise premium/discount.

    Args:
        df (pd.DataFrame): DataFrame containing option and stock data.
        option_type_col (Optional[str]): Name of the column indicating option type ('C' or 'P').
                                         If None, assumes call options for Delta/Theta/Rho calculations
                                         that differ by type.

    Returns:
        pd.DataFrame: DataFrame with added Greek columns (delta, gamma, vega, theta, rho).
                      Returns NaNs where calculation is not possible.
    """
    logging.info(f"Calculating approximate Greeks (Black-Scholes) for {len(df)} options...")
    result_df = df.copy()

    required_cols = ['stockClose', 'strike', 'daysToExpiry', 'risk_free_rate', 'impliedVolatility']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logging.error(f"Missing required columns for Greek calculation: {missing}")
        # Add NaN columns for Greeks if inputs are missing
        for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
             result_df[greek] = np.nan
        return result_df

    # Prepare inputs
    S = result_df['stockClose']
    K = result_df['strike']
    # Convert days to years, handle potential non-numeric values
    T_days = pd.to_numeric(result_df['daysToExpiry'], errors='coerce')
    T = T_days / 365.0
    r = result_df['risk_free_rate'] # Assumed to be in decimal format
    sigma = result_df['impliedVolatility'] # Assumed to be in decimal format

    # Identify rows with valid inputs (non-NaN and T > 0, sigma > 0)
    valid_mask = (S.notna() & K.notna() & T.notna() & r.notna() & sigma.notna() &
                  (T > 0) & (sigma > 0))
    logging.info(f"Number of rows with valid inputs for Greek calculation: {valid_mask.sum()}")

    # Initialize Greek columns with NaN
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        result_df[greek] = np.nan

    if valid_mask.sum() == 0:
        logging.warning("No valid rows found for Greek calculation.")
        return result_df

    # Calculate d1 and d2 only for valid rows
    d1, d2 = calculate_d1_d2(S[valid_mask], K[valid_mask], T[valid_mask], r[valid_mask], sigma[valid_mask])

    # --- Calculate Greeks ---
    # Gamma (same for call and put)
    result_df.loc[valid_mask, 'gamma'] = N_prime(d1) / (S[valid_mask] * sigma[valid_mask] * np.sqrt(T[valid_mask]))

    # Vega (same for call and put)
    result_df.loc[valid_mask, 'vega'] = S[valid_mask] * N_prime(d1) * np.sqrt(T[valid_mask])

    # --- Option Type Specific Greeks ---
    is_call = pd.Series(True, index=df.index) # Default to call
    if option_type_col and option_type_col in result_df.columns:
         # Ensure the column exists before using it
        is_call = result_df[option_type_col].str.upper() == 'C'
        logging.info(f"Identified {is_call.sum()} calls and {(~is_call).sum()} puts based on column '{option_type_col}'.")
    else:
        logging.warning(f"Option type column '{option_type_col}' not found or not specified. Calculating Delta, Theta, Rho assuming CALL options.")

    # Delta
    delta_values = N(d1) # Call delta
    # Apply put delta where applicable
    delta_values[~is_call[valid_mask]] = delta_values[~is_call[valid_mask]] - 1.0 # Put delta = N(d1) - 1
    result_df.loc[valid_mask, 'delta'] = delta_values

    # Theta
    theta_val1 = - (S[valid_mask] * N_prime(d1) * sigma[valid_mask]) / (2 * np.sqrt(T[valid_mask]))
    # Call theta = val1 - r * K * exp(-rT) * N(d2)
    call_theta = theta_val1 - r[valid_mask] * K[valid_mask] * np.exp(-r[valid_mask] * T[valid_mask]) * N(d2)
    # Put theta = val1 + r * K * exp(-rT) * N(-d2)
    put_theta = theta_val1 + r[valid_mask] * K[valid_mask] * np.exp(-r[valid_mask] * T[valid_mask]) * N(-d2)

    theta_values = call_theta.copy()
    theta_values[~is_call[valid_mask]] = put_theta[~is_call[valid_mask]]
    # Theta is typically annualized, convert to per-day by dividing by 365
    result_df.loc[valid_mask, 'theta'] = theta_values / 365.0

    # Rho
    # Call rho = K * T * exp(-rT) * N(d2)
    call_rho = K[valid_mask] * T[valid_mask] * np.exp(-r[valid_mask] * T[valid_mask]) * N(d2)
    # Put rho = -K * T * exp(-rT) * N(-d2)
    put_rho = -K[valid_mask] * T[valid_mask] * np.exp(-r[valid_mask] * T[valid_mask]) * N(-d2)

    rho_values = call_rho.copy()
    rho_values[~is_call[valid_mask]] = put_rho[~is_call[valid_mask]]
    # Rho is typically per 1% change in interest rate, so divide by 100
    result_df.loc[valid_mask, 'rho'] = rho_values / 100.0

    # Log counts of calculated Greeks
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        calculated_count = result_df[greek].notna().sum()
        logging.info(f"Calculated {greek} for {calculated_count} options.")
        if calculated_count < valid_mask.sum():
             logging.warning(f"Could not calculate {greek} for {valid_mask.sum() - calculated_count} options with valid inputs (check for intermediate NaNs or Infs).")

    return result_df

def add_risk_free_rate(df: pd.DataFrame, rate_file: str, date_col: str = 'quoteDate') -> pd.DataFrame:
    """
    Loads risk-free rate data, merges it with the main DataFrame, and interpolates missing values.

    Args:
        df (pd.DataFrame): Main options DataFrame.
        rate_file (str): Path to the CSV file containing risk-free rates (e.g., DGS10.csv).
                         Expected columns: 'DATE', 'DGS10' (or similar rate column).
        date_col (str): The date column in the main DataFrame to merge on.

    Returns:
        pd.DataFrame: DataFrame with 'risk_free_rate' column added and interpolated.
    """
    logging.info(f"Loading risk-free rates from: {rate_file}")
    try:
        rates_df = pd.read_csv(rate_file)
        # Find the rate column (assuming it's the second column if not named DGS10)
        rate_col_name = 'DGS10'
        if rate_col_name not in rates_df.columns:
             if len(rates_df.columns) > 1:
                  rate_col_name = rates_df.columns[1]
                  logging.warning(f"'DGS10' column not found in rate file. Using column '{rate_col_name}' instead.")
             else:
                  raise ValueError("Rate file does not contain a recognizable rate column.")

        logging.info(f"Using rate column: '{rate_col_name}'")

        # Rename columns for clarity and consistency
        rates_df = rates_df[['DATE', rate_col_name]].rename(columns={'DATE': 'rate_date', rate_col_name: 'rate_value'})

        # Convert date columns to datetime objects
        rates_df['rate_date'] = pd.to_datetime(rates_df['rate_date'])
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
             df[date_col] = pd.to_datetime(df[date_col])

        # Convert rate value to numeric, coercing errors (like '.') to NaN
        rates_df['rate_value'] = pd.to_numeric(rates_df['rate_value'], errors='coerce')

        # Convert rate from percentage to decimal
        rates_df['risk_free_rate'] = rates_df['rate_value'] / 100.0

        # Sort both dataframes by date before merging/interpolating
        df = df.sort_values(by=date_col)
        rates_df = rates_df.sort_values(by='rate_date')

        # Merge rates - use merge_asof for nearest date matching (allow backward fill)
        logging.info(f"Merging risk-free rates based on '{date_col}' using merge_asof.")
        merged_df = pd.merge_asof(
            df,
            rates_df[['rate_date', 'risk_free_rate']],
            left_on=date_col,
            right_on='rate_date',
            direction='backward' # Find the latest rate on or before the quote date
        )

        # Check how many rates were initially missing
        initial_missing = merged_df['risk_free_rate'].isna().sum()
        if initial_missing > 0:
            logging.warning(f"Found {initial_missing} rows with missing risk-free rates after merge_asof.")
            # Attempt interpolation (linear) - requires sorted data
            merged_df['risk_free_rate'] = merged_df['risk_free_rate'].interpolate(method='linear')
            final_missing = merged_df['risk_free_rate'].isna().sum()
            logging.info(f"Interpolated missing rates. Remaining missing: {final_missing}")
            if final_missing > 0:
                 # If still missing (likely at the beginning), forward fill
                 merged_df['risk_free_rate'] = merged_df['risk_free_rate'].ffill()
                 # And backward fill for any remaining at the very start
                 merged_df['risk_free_rate'] = merged_df['risk_free_rate'].bfill()
                 logging.info(f"Applied ffill/bfill. Final missing rates: {merged_df['risk_free_rate'].isna().sum()}")
        else:
             logging.info("No missing risk-free rates after merge_asof.")

        # Drop the temporary rate_date column
        if 'rate_date' in merged_df.columns:
             merged_df = merged_df.drop(columns=['rate_date'])

        return merged_df

    except FileNotFoundError:
        logging.error(f"Risk-free rate file not found: {rate_file}")
        raise
    except Exception as e:
        logging.error(f"Error processing risk-free rate file: {str(e)}")
        # Add NaN column if processing fails
        df['risk_free_rate'] = np.nan
        return df