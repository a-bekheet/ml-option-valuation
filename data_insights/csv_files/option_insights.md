Here's the column analysis summary in a proper table format:

| Column Name        | Data Type          | Percentage Present | Percentage Unique | Recommendation |
|--------------------|--------------------|--------------------|-------------------|----------------|
| contractSymbol     | category           | 100.00             | 2.08              | Keep           |
| lastTradeDate      | datetime64[ns, UTC]| 100.00             | 21.68             | Keep           |
| strike             | float64            | 100.00             | 0.02              | Keep           |
| lastPrice          | float64            | 100.00             | 0.91              | Keep           |
| bid                | float64            | 99.60              | 0.47              | Keep           |
| ask                | float64            | 99.65              | 0.47              | Keep           |
| change             | float64            | 100.00             | 1.34              | Keep           |
| percentChange      | float64            | 99.40              | 12.44             | Review         |
| volume             | uint32             | 100.00             | 0.26              | Keep           |
| openInterest       | uint32             | 100.00             | 0.56              | Keep           |
| impliedVolatility  | float64            | 100.00             | 3.59              | Keep           |
| inTheMoney         | bool               | 100.00             | 0.00              | Keep           |
| contractSize       | category           | 100.00             | 0.00              | Keep           |
| currency           | category           | 99.99              | 0.00              | Keep           |
| quoteDate          | datetime64[ns]     | 100.00             | 0.01              | Keep           |
| expiryDate         | datetime64[ns]     | 100.00             | 0.00              | Keep           |
| daysToExpiry       | uint16             | 100.00             | 0.01              | Keep           |
| stockVolume        | uint32             | 100.00             | 0.10              | Keep           |
| stockClose         | float64            | 100.00             | 0.08              | Keep           |
| stockAdjClose      | float64            | 100.00             | 0.12              | Keep           |
| stockOpen          | float64            | 100.00             | 0.08              | Keep           |
| stockHigh          | float64            | 100.00             | 0.08              | Keep           |
| stockLow           | float64            | 100.00             | 0.08              | Keep           |
| strikeDelta        | float64            | 100.00             | 4.67              | Keep           |
| stockClose_ewm_5d  | float64            | 100.00             | 0.11              | Keep           |
| stockClose_ewm_15d | float64            | 100.00             | 0.11              | Keep           |
| stockClose_ewm_45d | float64            | 100.00             | 0.11              | Keep           |
| stockClose_ewm_135d| float64            | 100.00             | 0.11              | Keep           |
| ticker             | object             | 100.00             | 0.00              | Keep           |

The table includes the following columns:
- Column Name: The name of each column in the dataset.
- Data Type: The data type of each column.
- Percentage Present: The percentage of non-null values in each column.
- Percentage Unique: The percentage of unique values in each column.
- Recommendation: The recommendation for each column (either "Keep" or "Review").