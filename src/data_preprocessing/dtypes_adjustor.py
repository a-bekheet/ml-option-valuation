import pandas as pd
import time

def process_full_dataset(input_file: str, output_file: str, chunksize: int = 100000) -> None:
    total_rows = sum(1 for _ in open(input_file)) - 1
    dtypes = {
        'strike': 'float32',
        'lastPrice': 'float32',
        'bid': 'float32',
        'ask': 'float32',
        'change': 'float32',
        'percentChange': 'float32',
        'impliedVolatility': 'float32',
        'contractSize': 'category',
        'currency': 'category',
        'stockClose': 'float32',
        'stockHigh': 'float32',
        'stockLow': 'float32',
        'stockOpen': 'float32',
        'stockVolume': 'float32',
        'volume': 'float32',
        'openInterest': 'float32'
    }
    date_columns = ['lastTradeDate', 'quoteDate', 'expiryDate']
    
    def convert_types(chunk: pd.DataFrame) -> pd.DataFrame:
        for col in ['stockVolume', 'volume', 'openInterest']:
            chunk[col] = pd.to_numeric(chunk[col], downcast='unsigned')
        chunk[date_columns] = chunk[date_columns].apply(pd.to_datetime)
        chunk['daysToExpiry'] = pd.to_numeric(
            chunk['daysToExpiry'].str.replace(' days', ''),
            downcast='unsigned'
        )
        return chunk
    
    first_chunk = True
    rows_processed = 0
    try:
        for chunk in pd.read_csv(input_file, chunksize=chunksize, dtype=dtypes):
            chunk = convert_types(chunk)
            chunk.to_csv(
                output_file,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            rows_processed += len(chunk)
            if first_chunk:
                first_chunk = False
            if rows_processed % (chunksize * 10) == 0:
                percent_complete = (rows_processed / total_rows) * 100
                print(f"Processed: {rows_processed:,}/{total_rows:,} rows ({percent_complete:.1f}%)")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise