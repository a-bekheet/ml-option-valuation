# src/data_preprocessing/greeks_calculator.py

import traceback
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
    Handles variations in column names for DATE and RATE columns in the rate file.

    Args:
        df (pd.DataFrame): Main options DataFrame.
        rate_file (str): Path to the CSV file containing risk-free rates (e.g., DGS10.csv).
        date_col (str): The date column in the main DataFrame to merge on.

    Returns:
        pd.DataFrame: DataFrame with 'risk_free_rate' column added and interpolated.
    """
    logging.info(f"Attempting to load risk-free rates from: {rate_file}")
    try:
        rates_df = pd.read_csv(rate_file)
        logging.info(f"Successfully loaded rate file. Columns found: {rates_df.columns.tolist()}")

        if rates_df.empty:
            raise ValueError("Risk-free rate file is empty.")
        if len(rates_df.columns) < 2:
            raise ValueError(f"Risk-free rate file expected at least 2 columns, found {len(rates_df.columns)}.")

        # --- More Robust Column Identification ---
        date_col_name_in_rates_file = None
        rate_col_name_in_rates_file = None
        available_cols = [col.strip() for col in rates_df.columns] # Strip spaces from names

        # Try finding Date Column (case-insensitive, check specific names first)
        potential_date_names = ['observation_date', 'DATE', 'Date', 'date']
        for name in potential_date_names:
            for col in available_cols:
                if col.upper() == name.upper():
                    date_col_name_in_rates_file = col
                    logging.info(f"Identified date column: '{date_col_name_in_rates_file}'")
                    break
            if date_col_name_in_rates_file: break

        # Fallback to first column if not found by name
        if not date_col_name_in_rates_file:
             date_col_name_in_rates_file = available_cols[0]
             logging.warning(f"Using first column '{date_col_name_in_rates_file}' as date column.")

        # Try finding Rate Column (case-insensitive, check specific names first)
        potential_rate_names = ['DGS10', 'Rate', 'Value', 'rate_value']
        for name in potential_rate_names:
             for col in available_cols:
                 # Ensure it's not the date column we already found
                 if col.upper() == name.upper() and col != date_col_name_in_rates_file:
                     rate_col_name_in_rates_file = col
                     logging.info(f"Identified rate column: '{rate_col_name_in_rates_file}'")
                     break
             if rate_col_name_in_rates_file: break

        # Fallback to second column if not found by name
        if not rate_col_name_in_rates_file:
             if len(available_cols) > 1 and available_cols[1] != date_col_name_in_rates_file:
                 rate_col_name_in_rates_file = available_cols[1]
                 logging.warning(f"Using second column '{rate_col_name_in_rates_file}' as rate column.")
             else:
                 raise ValueError("Could not reliably identify distinct date and rate columns in the rate file.")

        # --- End Column Identification ---

        # **Debug Log**: Log identified names before selection
        logging.debug(f"Attempting to select using date='{date_col_name_in_rates_file}' and rate='{rate_col_name_in_rates_file}'")

        # Select and rename using the identified column names
        rates_df_renamed = rates_df[[date_col_name_in_rates_file, rate_col_name_in_rates_file]].rename(columns={
            date_col_name_in_rates_file: 'rate_date',
            rate_col_name_in_rates_file: 'rate_value'
        })

        # Convert date columns to datetime objects
        rates_df_renamed['rate_date'] = pd.to_datetime(rates_df_renamed['rate_date'], errors='coerce')
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
             df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Drop rows where date conversion failed
        rates_df_renamed = rates_df_renamed.dropna(subset=['rate_date'])
        df = df.dropna(subset=[date_col])
        logging.debug(f"Rows after dropping invalid dates - Rates DF: {len(rates_df_renamed)}, Main DF: {len(df)}")


        # Convert rate value to numeric, coercing errors (like '.') to NaN
        rates_df_renamed['rate_value'] = pd.to_numeric(rates_df_renamed['rate_value'], errors='coerce')
        nan_rates_count = rates_df_renamed['rate_value'].isna().sum()
        if nan_rates_count > 0:
            logging.warning(f"Found {nan_rates_count} non-numeric rate values (e.g., '.') in rate file. Coerced to NaN and dropping these rows.")
            rates_df_renamed = rates_df_renamed.dropna(subset=['rate_value']) # Drop rows with NaN rates before processing

        # Check if rates_df is empty after dropping NaNs
        if rates_df_renamed.empty:
            raise ValueError("Rate data became empty after handling invalid dates/values.")

        # Convert rate from percentage to decimal
        max_rate = rates_df_renamed['rate_value'].max()
        if max_rate > 1.0: # Check if rates look like percentages
             logging.info(f"Max rate value is {max_rate}. Assuming rates are percentages, dividing by 100.")
             rates_df_renamed['risk_free_rate'] = rates_df_renamed['rate_value'] / 100.0
        else:
             logging.info(f"Max rate value is {max_rate}. Assuming rates are already decimals.")
             rates_df_renamed['risk_free_rate'] = rates_df_renamed['rate_value']

        # Sort both dataframes by date before merging
        df = df.sort_values(by=date_col)
        rates_df_renamed = rates_df_renamed.sort_values(by='rate_date').drop_duplicates(subset=['rate_date'], keep='first')

        # Merge rates using merge_asof
        logging.info(f"Merging risk-free rates onto main DataFrame using {date_col}...")
        merged_df = pd.merge_asof(
            df,
            rates_df_renamed[['rate_date', 'risk_free_rate']],
            left_on=date_col,
            right_on='rate_date',
            direction='backward',
            tolerance=pd.Timedelta(days=7) # Look back up to 7 days for a rate
        )
        logging.debug(f"Shape after merge_asof: {merged_df.shape}")

        # Interpolate and fill remaining missing rates
        initial_missing = merged_df['risk_free_rate'].isna().sum()
        if initial_missing > 0:
            logging.warning(f"{initial_missing} rows initially missing risk-free rate after merge_asof.")
            logging.info("Attempting linear interpolation...")
            merged_df['risk_free_rate'] = merged_df['risk_free_rate'].interpolate(method='linear')
            interpolated_missing = merged_df['risk_free_rate'].isna().sum()
            logging.info(f"Missing after interpolation: {interpolated_missing}")
            if interpolated_missing > 0:
                 logging.info("Applying ffill and bfill for remaining gaps...")
                 merged_df['risk_free_rate'] = merged_df['risk_free_rate'].ffill()
                 merged_df['risk_free_rate'] = merged_df['risk_free_rate'].bfill()
                 final_missing = merged_df['risk_free_rate'].isna().sum()
                 logging.info(f"Final missing rates: {final_missing}")
                 if final_missing > 0:
                      # If still missing, maybe fill with a global average or a default?
                      global_avg_rate = merged_df['risk_free_rate'].mean() # Calculate average from non-missing
                      logging.warning(f"Filling {final_missing} remaining missing rates with global average: {global_avg_rate:.4f}")
                      merged_df['risk_free_rate'].fillna(global_avg_rate, inplace=True)
                      # Or raise an error if rates are critical:
                      # raise ValueError(f"{final_missing} risk-free rates could not be determined.")
        else:
             logging.info("No missing risk-free rates found after merge_asof.")

        # Drop the temporary rate_date column
        if 'rate_date' in merged_df.columns:
             merged_df = merged_df.drop(columns=['rate_date'])

        logging.info("Finished adding risk-free rate.")
        return merged_df

    except FileNotFoundError:
        logging.error(f"Risk-free rate file not found: {rate_file}")
        raise
    except Exception as e:
        logging.error(f"Error processing risk-free rate file '{rate_file}': {str(e)}")
        logging.error(traceback.format_exc())
        # Add NaN column and return if error occurs, allows pipeline to potentially continue
        logging.warning("Adding 'risk_free_rate' column with NaNs due to processing error.")
        df['risk_free_rate'] = np.nan
        return df