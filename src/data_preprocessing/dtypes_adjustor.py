import logging
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def datatype_adjustor(input_file: str, output_file: str, chunksize: int = 100000) -> None:
    """
    Adjusts data types for specified columns in a CSV or NPY file.
    Handles large CSV files using chunking. Loads NPY files entirely.

    Args:
        input_file (str): Path to the input CSV or NPY file.
        output_file (str): Path to save the processed CSV file.
        chunksize (int): Rows per chunk for CSV processing (ignored for NPY).
    """
    logging.info(f"Starting data type adjustment for: {input_file}")
    start_time = time.time()

    # Define type conversions and date columns
    dtypes_to_convert = {
        'strike': 'float32', 'lastPrice': 'float32', 'bid': 'float32',
        'ask': 'float32', 'change': 'float32', 'percentChange': 'float32',
        'impliedVolatility': 'float32', 'contractSize': 'category',
        'currency': 'category', 'stockClose': 'float32', 'stockHigh': 'float32',
        'stockLow': 'float32', 'stockOpen': 'float32', 'stockAdjClose': 'float32', # Added stockAdjClose based on pipeline
        # Use pandas to_numeric for these as downcast='unsigned' might fail on NaN/large values initially
        'stockVolume': 'float64', # Load as float first, then maybe downcast if appropriate
        'volume': 'float64',
        'openInterest': 'float64'
    }
    # Columns to attempt unsigned integer conversion *after* loading as float
    uint_cols = ['stockVolume', 'volume', 'openInterest']
    date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']

    def convert_chunk_types(chunk: pd.DataFrame) -> pd.DataFrame:
        """Applies type conversions to a DataFrame chunk."""
        # Convert specific float columns
        for col, dtype in dtypes_to_convert.items():
             if col in chunk.columns and col not in uint_cols: # Exclude uint candidates for now
                 try:
                     if dtype == 'category':
                          chunk[col] = chunk[col].astype('category')
                     else:
                         # Use pd.to_numeric for robustness, then cast
                         chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype(dtype)
                 except Exception as e:
                     logging.warning(f"Could not convert column '{col}' to {dtype}: {e}")

        # Handle potential uint columns
        for col in uint_cols:
            if col in chunk.columns:
                try:
                    # Convert to float first to handle potential non-numeric strings, then try uint
                    numeric_series = pd.to_numeric(chunk[col], errors='coerce')
                    # Attempt downcast to unsigned integer, checking bounds
                    # Fill NaN temporarily, check min, then apply downcast where possible
                    min_val = numeric_series.min()
                    if pd.notna(min_val) and min_val >= 0:
                         # Use appropriate unsigned type based on max value if needed, or just uint64/uint32
                         # pd.to_numeric with downcast='unsigned' is often sufficient if no NaNs remain
                         # Let's keep them as float32/64 for now to avoid issues with NaNs during ML steps
                         # Or handle NaNs appropriately if integer type is strictly required
                         chunk[col] = numeric_series.astype('float32') # Keep as float32 is safer
                         # chunk[col] = numeric_series.fillna(-1).astype(np.uint32) # Example if filling NaN
                    else:
                         # Keep as float if negative values or all NaN
                         chunk[col] = numeric_series.astype('float32')

                except Exception as e:
                    logging.warning(f"Could not convert column '{col}' to numeric/uint: {e}")
                    # Ensure it's at least numeric if possible
                    if col in chunk.columns:
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')


        # Convert date columns
        for col in date_columns:
            if col in chunk.columns:
                try:
                     # Handle potential variations in date format if needed
                     chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
                except Exception as e:
                     logging.warning(f"Could not convert column '{col}' to datetime: {e}")

        # Convert 'daysToExpiry'
        if 'daysToExpiry' in chunk.columns:
             try:
                 # Check if it's already numeric
                 if pd.api.types.is_numeric_dtype(chunk['daysToExpiry']):
                     # Ensure it's integer-like, handle potential floats from NPY load
                     chunk['daysToExpiry'] = pd.to_numeric(chunk['daysToExpiry'], errors='coerce').round().astype('Int32') # Use nullable Int32
                 else:
                     # Assume string like '141 days' if not numeric
                     numeric_days = pd.to_numeric(chunk['daysToExpiry'].astype(str).str.replace(r'\s*days', '', regex=True), errors='coerce')
                     chunk['daysToExpiry'] = numeric_days.astype('Int32') # Use nullable Int32
             except Exception as e:
                 logging.warning(f"Could not convert column 'daysToExpiry': {e}")
                 chunk['daysToExpiry'] = pd.to_numeric(chunk['daysToExpiry'], errors='coerce').astype('Int32')


        # Convert 'inTheMoney' (specific to options data)
        if 'inTheMoney' in chunk.columns:
            try:
                # Handle boolean or string representations
                if chunk['inTheMoney'].dtype == 'bool':
                    chunk['inTheMoney'] = chunk['inTheMoney'].astype('boolean') # Use nullable boolean
                else: # Handle strings 'True'/'False' or similar if needed
                    chunk['inTheMoney'] = chunk['inTheMoney'].astype(str).str.lower().map({'true': True, 'false': False})
                    chunk['inTheMoney'] = chunk['inTheMoney'].astype('boolean') # Use nullable boolean
            except Exception as e:
                 logging.warning(f"Could not convert column 'inTheMoney' to boolean: {e}")


        return chunk

    # --- Processing Logic ---
    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext == '.npy':
        logging.info(f"Processing NPY file: {input_file}")
        try:
            # Define column names expected for NPY files
            # Ensure this matches the actual structure or adapt as needed
            npy_column_names = [
                'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice',
                'bid', 'ask', 'change', 'percentChange', 'volume',
                'openInterest', 'impliedVolatility', 'inTheMoney',
                'contractSize', 'currency', 'quoteDate', 'expiryDate',
                'daysToExpiry', 'stockVolume', 'stockClose', 'stockAdjClose',
                'stockOpen', 'stockHigh', 'stockLow', 'strikeDelta',
                'stockClose_ewm_5d', 'stockClose_ewm_15d', 'stockClose_ewm_45d',
                'stockClose_ewm_135d'
            ]
            data = np.load(input_file, allow_pickle=True)
            logging.info(f"Loaded NPY array with shape {data.shape}")

            # Create DataFrame - Adjust columns if shape mismatch
            if data.shape[1] == len(npy_column_names):
                df = pd.DataFrame(data, columns=npy_column_names)
            else:
                logging.warning(f"NPY column count mismatch: Got {data.shape[1]}, expected {len(npy_column_names)}. Adjusting columns.")
                if data.shape[1] < len(npy_column_names):
                    cols = npy_column_names[:data.shape[1]]
                else:
                    cols = npy_column_names + [f"Unknown_{i}" for i in range(data.shape[1] - len(npy_column_names))]
                df = pd.DataFrame(data, columns=cols)

            logging.info("Applying type conversions to NPY data...")
            df_converted = convert_chunk_types(df) # Apply to the whole dataframe
            logging.info("Saving adjusted data to CSV...")
            df_converted.to_csv(output_file, index=False)
            total_rows = len(df_converted)
            logging.info(f"Processed {total_rows:,} rows from NPY file.")

        except Exception as e:
            logging.error(f"Error processing NPY file {input_file}: {e}")
            raise

    elif file_ext == '.csv':
        logging.info(f"Processing CSV file: {input_file} in chunks")
        total_rows_processed = 0
        first_chunk = True
        try:
            # Estimate total rows for progress (optional, might fail on very large files)
            try:
                 total_rows = sum(1 for _ in open(input_file, encoding='utf-8')) - 1 # Assume header
                 logging.info(f"Estimated total rows: {total_rows:,}")
                 use_tqdm = True
            except Exception as count_err:
                 logging.warning(f"Could not estimate total rows for progress bar: {count_err}. Progress will be shown per chunk.")
                 total_rows = None
                 use_tqdm = False

            # Define dtypes for faster CSV reading (use float64 for uint candidates initially)
            read_dtypes = dtypes_to_convert.copy()
            for col in uint_cols:
                 if col in read_dtypes: read_dtypes[col] = 'object' # Read as object initially to handle potential bad data before numeric conversion


            iterator = pd.read_csv(
                input_file,
                chunksize=chunksize,
                dtype=read_dtypes, # Apply basic dtypes on read
                parse_dates=False, # Handle dates manually in convert_chunk_types
                low_memory=False   # Recommended for mixed types / large files
            )

            # Setup tqdm progress bar if total_rows is known
            progress_bar = tqdm(iterator, total=(total_rows // chunksize + 1) if total_rows else None, unit="chunk", disable=not use_tqdm)

            for chunk in progress_bar:
                if use_tqdm: progress_bar.set_description("Processing chunk")

                chunk_converted = convert_chunk_types(chunk) # Apply detailed conversions
                chunk_converted.to_csv(
                    output_file,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False
                )
                total_rows_processed += len(chunk)
                if first_chunk:
                    first_chunk = False
                # Log progress less frequently
                # if total_rows_processed % (chunksize * 5) == 0: # Log every 5 chunks
                #     percent_complete = (total_rows_processed / total_rows * 100) if total_rows else 'N/A'
                #     logging.info(f"Processed: {total_rows_processed:,} rows ({percent_complete:.1f}%)")


            logging.info(f"Finished processing CSV. Total rows processed: {total_rows_processed:,}")

        except Exception as e:
            logging.error(f"Error processing CSV file {input_file} at row approx {total_rows_processed}: {e}")
            raise
    else:
         raise ValueError(f"Unsupported file type: {file_ext}. Only .csv and .npy are supported.")

    end_time = time.time()
    logging.info(f"Data type adjustment finished in {end_time - start_time:.2f} seconds. Output: {output_file}")