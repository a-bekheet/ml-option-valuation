#!/bin/bash

# Script to run tests on the sampled options data

# Create output directory
mkdir -p test_results_sampled

# Run tests on the entire sampled dataset
echo "Running tests on full sampled dataset..."
python run-tests.py \
  --raw sampled_data/option_data_with_headers_sample_99974rows_20250310_173210.csv \
  --output-dir test_results_sampled

# Optional: Run tests on specific tickers
# Uncomment these lines to test specific tickers

# echo "Running tests on AAPL ticker..."
# python run-tests.py \
#   --raw sampled_data/option_data_with_headers_sample_99974rows_20250310_173210.csv \
#   --output-dir test_results_sampled/AAPL \
#   --ticker AAPL

# echo "Running tests on TSLA ticker..."
# python run-tests.py \
#   --raw sampled_data/option_data_with_headers_sample_99974rows_20250310_173210.csv \
#   --output-dir test_results_sampled/TSLA \
#   --ticker TSLA

echo "Tests completed! See test_results_sampled directory for detailed results."