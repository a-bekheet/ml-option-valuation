# Option ML Prediction

This repository contains a machine learning project focused on predicting options prices. The project leverages various machine learning algorithms and financial data to make accurate predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/option-ml-prediction.git
cd option-ml-prediction
pip install -r requirements.txt
```

## Usage

To run the prediction model, use the following command still unimplemented:

```bash
python predict.py --input data/options_data.csv --output results/predictions.csv
```

## Project Structure

```
option-ml-prediction/
├── data_files/             # Dataset files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_insights/                      # Data sampling and statistics
│   ├── data_presence_cleaning/             # Data Preprocessing and Manipulation
│   │   ├── adjusting_mislabelled_data/     # Orthogonalization and Relabelling data
│   │   ├── fixing_datatypes/               # Storage reduction
│   │   ├── fixing_stock_splits/            # Stock split identification and adjustments
│   │   ├── greek_calculating/              # Second-Order Derivatives
│   │   ├── removing_stockClose/            # Dropping Data based on statistics
├── requirements.txt        # Python dependencies
└── README.md               # Project README
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.