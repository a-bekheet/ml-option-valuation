import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import List, Dict, Any

def verify_column_presence(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """
    Checks whether all expected columns are present in the dataframe.
    Raises a ValueError if any are missing.
    """
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def verify_boolean_column(df: pd.DataFrame, col_name: str) -> None:
    """
    Ensures that a column meant to be boolean (0/1) has only valid values.
    Raises a ValueError if invalid values are found.
    """
    if col_name not in df.columns:
        return  # Skip if the column doesn't exist
    unique_vals = df[col_name].dropna().unique()
    # valid values are {0, 1}
    invalid_vals = [v for v in unique_vals if v not in [0, 1]]
    if invalid_vals:
        raise ValueError(
            f"Column '{col_name}' has invalid boolean values: {invalid_vals}"
        )


def verify_cyclical_range(df: pd.DataFrame, cyclical_cols: List[str]) -> None:
    """
    Checks that each cyclical column is strictly within the expected range [-1, 1].
    Raises a ValueError if out-of-bounds values are detected.
    """
    for col in cyclical_cols:
        if col not in df.columns:
            continue
        min_val = df[col].min(skipna=True)
        max_val = df[col].max(skipna=True)
        if min_val < -1.000001 or max_val > 1.000001:
            raise ValueError(
                f"Cyclical column '{col}' out of expected range [-1, 1]: "
                f"(min={min_val}, max={max_val})"
            )


def aggregate_stats(df: pd.DataFrame, stats_dict: Dict[str, Dict[str, Any]], numeric_cols: List[str]) -> None:
    """
    Update the stats_dict with aggregated values (sum, sum of squares, min, max, count)
    for each column in numeric_cols from the given dataframe chunk.
    """
    for col in numeric_cols:
        # ignore if column doesn't exist in this chunk
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        
        cmin = col_data.min()
        cmax = col_data.max()
        csum = col_data.sum()
        csum_sq = np.sum(col_data**2)  # sum of squares
        ccount = col_data.shape[0]
        
        if col not in stats_dict:
            stats_dict[col] = {
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": np.inf,
                "max": -np.inf,
                "count": 0
            }
        
        stats_dict[col]["sum"] += csum
        stats_dict[col]["sum_sq"] += csum_sq
        if cmin < stats_dict[col]["min"]:
            stats_dict[col]["min"] = cmin
        if cmax > stats_dict[col]["max"]:
            stats_dict[col]["max"] = cmax
        stats_dict[col]["count"] += ccount


def finalize_stats(stats_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Once all chunks are processed, compute mean, std, min, max for each column
    based on aggregated sums and sums of squares.
    Returns a DataFrame with final stats.
    """
    rows = []
    for col, agg in stats_dict.items():
        total_count = agg["count"]
        if total_count == 0:
            # no data for this column
            rows.append([col, np.nan, np.nan, np.nan, np.nan, 0])
            continue
        
        # mean
        mean = agg["sum"] / total_count
        # variance
        variance = (agg["sum_sq"] / total_count) - (mean**2)
        std = np.sqrt(variance) if variance >= 0 else np.nan
        
        rows.append([
            col,
            mean,
            std,
            agg["min"],
            agg["max"],
            total_count
        ])
    df_summary = pd.DataFrame(rows, columns=["column", "mean", "std", "min", "max", "count"])
    return df_summary


def verify_scaled_file(
    file_path: str,
    chunksize: int = 200_000,
    expected_numeric_cols: List[str] = None,
    cyclical_cols: List[str] = None,
    bool_cols: List[str] = None
) -> pd.DataFrame:
    """
    Reads the final scaled CSV in chunks and verifies:
      1) All expected columns are present.
      2) Boolean columns are strictly 0/1.
      3) Cyclical columns remain in [-1, 1].
      4) Basic stats (mean, std, min, max) across numeric columns.
    Returns a DataFrame with final summary statistics for numeric columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # We will gather aggregator stats across all chunks
    stats_dict = {}
    
    # We iterate once to check presence, types, boolean columns, cyc ranges, etc.
    # We'll track row_count for a progress bar
    total_rows = 0
    # Attempt to get total lines for progress bar
    # (May be large, but we do this for a rough progress estimate)
    try:
        # +1 if there's a header
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        total_rows = max(total_lines - 1, 0)
    except:
        total_rows = None  # fallback if the file is extremely large
    
    print("Starting verification of scaled file:", file_path)
    
    row_counter = 0
    # Create a progress bar if we can estimate total_rows
    with pd.read_csv(file_path, chunksize=chunksize) as reader:
        for chunk in tqdm(reader, total= None if not total_rows else (total_rows // chunksize + 1), unit="chunk"):
            # 1) Verify column presence if we have expected columns
            if expected_numeric_cols:
                verify_column_presence(chunk, expected_numeric_cols)
            
            # 2) Verify boolean columns are 0/1
            if bool_cols:
                for bcol in bool_cols:
                    verify_boolean_column(chunk, bcol)
            
            # 3) Verify cyclical columns in [-1, 1]
            if cyclical_cols:
                verify_cyclical_range(chunk, cyclical_cols)
            
            # 4) Aggregate stats for numeric columns
            if expected_numeric_cols:
                aggregate_stats(chunk, stats_dict, expected_numeric_cols)
            
            row_counter += len(chunk)
    
    print(f"\nFinished reading {row_counter} rows from '{file_path}'")
    
    # Produce final stats DataFrame
    df_summary = finalize_stats(stats_dict)
    return df_summary


def main():
    # Adjust these to match your scenario:
    final_scaled_file = "/Users/bekheet/dev/option-ml-prediction/data_files/option_data_scaled.csv"
    
    # Columns we EXPLICITLY expect to be numeric and scaled.
    # (If you prefer to gather them dynamically, you can also do so, but being explicit is safer.)
    numeric_cols = [
        # Example of typical numeric columns:
        "strike", "lastPrice", "bid", "ask", "change", "percentChange",
        "volume", "openInterest", "impliedVolatility", "daysToExpiry",
        "stockVolume", "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
        "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
        "stockClose_ewm_45d", "stockClose_ewm_135d",
        "day_of_week", "day_of_month", "day_of_year"
        # etc...
    ]
    
    # Columns we expect to remain cyclical and unscaled (still in [-1,1])
    cyclical_cols = [
        "day_of_week_sin", "day_of_week_cos",
        "day_of_month_sin", "day_of_month_cos",
        "day_of_year_sin", "day_of_year_cos"
    ]
    
    # Columns that should remain 0/1 (binary)
    bool_cols = [
        "inTheMoney"
    ]
    
    # Perform verification
    print("Verifying final scaled CSV...")
    df_summary = verify_scaled_file(
        file_path=final_scaled_file,
        chunksize=200_000,
        expected_numeric_cols=numeric_cols,
        cyclical_cols=cyclical_cols,
        bool_cols=bool_cols
    )
    
    print("\nVERIFICATION SUMMARY OF NUMERIC COLUMNS:")
    print(df_summary.to_string(index=False))
    
    # Optional: further checks on means, std, etc.
    # Example threshold check:
    #   - For standard scaling, you might expect means to be ~0,
    #     std to be ~1. We'll just print them here.
    
    # If you want an automated pass/fail:
    out_of_range = df_summary[
        (df_summary["mean"].abs() > 0.05) |  # e.g., check if mean is within ±0.05
        (df_summary["std"] < 0.8) | (df_summary["std"] > 1.2)  # e.g., check if std is between [0.8,1.2]
    ]
    if not out_of_range.empty:
        print("\nWARNING: Some columns deviate significantly from mean=0, std=1:")
        print(out_of_range.to_string(index=False))
    else:
        print("\nAll columns appear within the expected standard-scaling range!")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
