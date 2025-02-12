import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from pathlib import Path

from utils.data_utils import get_available_tickers, select_ticker, StockOptionDataset
from utils.model_utils import (
    train_model, analyze_model_architecture, load_model,
    run_existing_model
)
from utils.menu_utils import display_menu
from utils.visualization_utils import save_and_display_results, display_model_analysis

class ImprovedMixedRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size_lstm=64, hidden_size_gru=64, num_layers=2, output_size=1):
        super(ImprovedMixedRNNModel, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.mid_bn = nn.BatchNorm1d(hidden_size_lstm)
        self.dropout = nn.Dropout(0.2)
        
        self.gru = nn.GRU(
            input_size=hidden_size_lstm,
            hidden_size=hidden_size_gru,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.bn_final = nn.BatchNorm1d(hidden_size_gru)
        self.fc1 = nn.Linear(hidden_size_gru, hidden_size_gru // 2)
        self.fc2 = nn.Linear(hidden_size_gru // 2, output_size)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x = x.view(-1, features)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, features)
        
        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out.contiguous()
        batch_size, seq_len, hidden_size = lstm_out.size()
        lstm_out = lstm_out.view(-1, hidden_size)
        lstm_out = self.mid_bn(lstm_out)
        lstm_out = lstm_out.view(batch_size, seq_len, hidden_size)
        lstm_out = self.dropout(lstm_out)
        
        gru_out, _ = self.gru(lstm_out)
        
        final_out = gru_out[:, -1, :]
        final_out = self.bn_final(final_out)
        final_out = self.dropout(final_out)
        final_out = torch.relu(self.fc1(final_out))
        final_out = self.fc2(final_out)
        
        return final_out

def train_option_model(data_path, ticker=None, seq_len=15, batch_size=128, epochs=20, 
                       hidden_size_lstm=128, hidden_size_gru=128, num_layers=2,
                       target_cols=["bid", "ask"]):
    """
    Train the option pricing model using the specified target columns.
    This version partitions the data in chronological order.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get available tickers and select one if not provided
    if ticker is None:
        tickers, counts = get_available_tickers(data_path)
        ticker = select_ticker(tickers, counts)
    
    # Initialize dataset (the CSV file should already be sorted temporally)
    dataset = StockOptionDataset(csv_file=data_path, ticker=ticker, seq_len=seq_len, target_cols=target_cols)
    
    if len(dataset) < 1:
        raise ValueError("Insufficient data for sequence creation!")
    
    # Instead of random_split, split the dataset by slicing indices to maintain temporal order
    total_len = len(dataset)
    train_len = int(0.80 * total_len)
    val_len = int(0.10 * total_len)
    test_len = total_len - train_len - val_len
    
    indices = list(range(total_len))
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len+val_len]
    test_indices = indices[train_len+val_len:]
    
    from torch.utils.data import Subset, DataLoader
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize model with output_size equal to the number of targets
    model = ImprovedMixedRNNModel(
        input_size=dataset.n_features,
        hidden_size_lstm=hidden_size_lstm,
        hidden_size_gru=hidden_size_gru,
        num_layers=num_layers,
        output_size=len(target_cols)
    )
    
    # Analyze model architecture
    model_analysis = analyze_model_architecture(
        model, 
        input_size=dataset.n_features,
        seq_len=seq_len
    )
    
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device
    )
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}, model_analysis, dataset.ticker, target_cols

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('option_model.log')
        ]
    )

def load_config():
    """Load and return the configuration settings."""
    return {
        'data_path': "data_files/option_data_scaled.csv",
        'seq_len': 15,
        'batch_size': 32,
        'epochs': 20,
        'hidden_size_lstm': 64,
        'hidden_size_gru': 64,
        'num_layers': 2,
        'ticker': None,
        'target_cols': ["bid", "ask"],
        'models_dir': "models"
    }

def validate_paths(config):
    """Validate and create necessary directories."""
    data_path = Path(config['data_path'])
    models_dir = Path(config['models_dir'])
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    models_dir.mkdir(exist_ok=True)
    return data_path, models_dir

def handle_train_model(config):
    """Handle the model training workflow."""
    try:
        logging.info("Starting model training...")
        model, history, analysis, ticker, target_cols = train_option_model(**config)
        save_and_display_results(model, history, analysis, ticker, target_cols)
        logging.info("Model training completed successfully")
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        print(f"\nError: {str(e)}")

def list_available_models(models_dir):
    """List and return available trained models."""
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        print("\nNo saved models found in", models_dir)
        return None
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    return model_files

def select_model(model_files):
    """Let user select a model from the list."""
    while True:
        try:
            model_choice = int(input("\nSelect a model number: "))
            if 1 <= model_choice <= len(model_files):
                return model_files[model_choice-1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nModel selection cancelled")
            return None

def handle_run_model(config, models_dir):
    """Handle the model prediction workflow."""
    try:
        model_files = list_available_models(models_dir)
        if not model_files:
            return

        selected_model = select_model(model_files)
        if not selected_model:
            return

        model_path = os.path.join(models_dir, selected_model)
        
        # Get available tickers and select one
        tickers, counts = get_available_tickers(config['data_path'])
        ticker = select_ticker(tickers, counts)
        
        # Create dataset for the selected ticker
        dataset = StockOptionDataset(
            csv_file=config['data_path'],
            ticker=ticker,
            target_cols=config['target_cols']
        )
        
        logging.info(f"Running predictions with model: {selected_model}")
        run_existing_model(
            model_path,
            ImprovedMixedRNNModel,
            dataset,
            target_cols=config['target_cols']
        )
        logging.info("Predictions completed successfully")
    except Exception as e:
        logging.error(f"Error during model prediction: {str(e)}")
        print(f"\nError: {str(e)}")

def handle_analyze_architecture(config):
    """Handle the model architecture analysis workflow."""
    try:
        print("\nAnalyzing network architecture...")
        model = ImprovedMixedRNNModel(
            input_size=23,
            hidden_size_lstm=config['hidden_size_lstm'],
            hidden_size_gru=config['hidden_size_gru'],
            num_layers=config['num_layers'],
            output_size=len(config['target_cols'])
        )
        analysis = analyze_model_architecture(model)
        display_model_analysis(analysis)
        logging.info("Architecture analysis completed")
    except Exception as e:
        logging.error(f"Error during architecture analysis: {str(e)}")
        print(f"\nError: {str(e)}")

def main():
    """Main application entry point with improved error handling and user experience."""
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting Option Trading Model application")
        
        # Load configuration
        config = load_config()
        
        # Validate paths
        try:
            data_path, models_dir = validate_paths(config)
        except FileNotFoundError as e:
            logging.error(str(e))
            print(f"\nError: {str(e)}")
            return
        
        while True:
            try:
                choice = display_menu()
                
                if choice == 1:
                    handle_train_model(config)
                elif choice == 2:
                    handle_run_model(config, models_dir)
                elif choice == 3:
                    handle_analyze_architecture(config)
                elif choice == 4:
                    print("\nExiting program...")
                    logging.info("Application terminated by user")
                    break
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                continue
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                print(f"\nAn unexpected error occurred: {str(e)}")
                continue
    
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        logging.info("Application terminated by user")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        print(f"\nA critical error occurred: {str(e)}")
    finally:
        logging.info("Application shutdown")

if __name__ == "__main__":
    main()