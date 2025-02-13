# Option ML Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project for predicting option prices using a hybrid LSTM-GRU neural network architecture. The model is designed to predict both bid and ask prices simultaneously, leveraging historical options data and various market indicators.

## Features

- 🚀 Hybrid LSTM-GRU architecture for improved sequence learning
- 📈 Multi-target prediction (bid and ask prices)
- 🔄 Real-time training visualization
- 📊 Comprehensive error metrics (MSE, RMSE, MAE, MAPE)
- 🎯 Early stopping and learning rate scheduling
- 💾 Model checkpointing and result visualization
- 🔍 Detailed architecture analysis tools

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Results Visualization](#results-visualization)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/option-ml-prediction.git
cd option-ml-prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
option-ml-prediction/
├── data_files/             # Dataset files
│   └── split_data/         # Ticker-specific data files
├── models/                 # Trained models and visualizations
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── nn.py             # Main neural network implementation
│   ├── utils/            # Utility modules
│   │   ├── data_utils.py        # Data loading and processing
│   │   ├── model_utils.py       # Model-related utilities
│   │   ├── menu_utils.py        # CLI menu interface
│   │   └── visualization_utils.py # Plotting and visualization
│   ├── data_insights/            # Data sampling and statistics
│   ├── data_preprocessing/       # Data preprocessing pipeline
│   │   ├── datatype_adjustment/  # Data type optimization
│   │   ├── feature_dropping/     # Feature selection
│   │   └── mislabelled_data_adjustment/ # Data cleaning
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Usage

### Training a New Model

```bash
python src/nn.py
```

Select option 1 from the menu to train a new model. You'll be prompted to:
1. Select a ticker from available options
2. Configure model parameters (or use defaults)
3. Monitor training progress
4. View final evaluation metrics

### Running Predictions with Existing Model

```bash
python src/nn.py
```

Select option 2 from the menu to:
1. Choose a previously trained model
2. Select a ticker for prediction
3. View prediction metrics

## Model Architecture

The model uses a hybrid architecture combining LSTM and GRU layers:

1. Input Layer with Batch Normalization
2. LSTM layers for initial sequence processing
3. GRU layers for refined feature extraction
4. Dense layers for final prediction
5. Multiple regularization techniques:
   - Dropout layers
   - Batch normalization
   - Weight decay
   - Gradient clipping

### Key Parameters

- Sequence Length: 15 time steps
- LSTM Hidden Size: 64-128 units
- GRU Hidden Size: 64-128 units
- Number of Layers: 2
- Learning Rate: 1e-3 with adaptive scheduling
- Batch Size: 32-128

## Data Preprocessing

The project includes comprehensive data preprocessing:

1. Feature Engineering:
   - Technical indicators
   - Moving averages
   - Temporal features

2. Data Cleaning:
   - Missing value handling
   - Outlier detection
   - Data type optimization

3. Feature Selection:
   - Correlation analysis
   - Feature importance ranking
   - Dimensionality reduction

## Training and Evaluation

### Training Process

- 80-10-10 temporal split (train-validation-test)
- Early stopping with patience=5
- Learning rate reduction on plateau
- Gradient clipping at 1.0
- MSE loss function

### Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

## Results Visualization

The project automatically generates:

1. Training curves showing:
   - Training loss
   - Validation loss
   - Learning rate changes

2. Model architecture analysis:
   - Parameter counts
   - Layer shapes
   - Memory usage

3. Prediction visualization:
   - Actual vs Predicted plots
   - Error distribution analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Maintain backwards compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Financial data providers
- Open source community for various tools and libraries used

---

For questions or support, please open an issue in the GitHub repository.
