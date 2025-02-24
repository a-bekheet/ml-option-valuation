import torch
import torch.nn as nn
import logging

from utils.data_utils import get_available_tickers, select_ticker, validate_paths, StockOptionDataset
from utils.model_utils import (
    train_model, analyze_model_architecture, load_model, 
    run_existing_model, calculate_errors, list_available_models, select_model,
    handle_train_model, handle_run_model, handle_analyze_architecture, handle_benchmark_architectures
)
from utils.menu_utils import display_menu, run_application_loop
from utils.visualization_utils import save_and_display_results, display_model_analysis
from utils.performance_utils import (
    track_performance, benchmark_architectures, generate_architecture_comparison,
    visualize_architectures, extended_train_model_with_tracking,
    calculate_directional_accuracy, calculate_max_error
)

# Model Architecture Classes
class HybridRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size_lstm=64, hidden_size_gru=64, num_layers=2, output_size=1):
        super(HybridRNNModel, self).__init__()
        
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

class GRUGRUModel(nn.Module):
    """GRU-GRU architecture for comparison"""
    def __init__(self, input_size, hidden_size_gru1=64, hidden_size_gru2=64, num_layers=2, output_size=1):
        super(GRUGRUModel, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size_gru1,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.mid_bn = nn.BatchNorm1d(hidden_size_gru1)
        self.dropout = nn.Dropout(0.2)
        
        self.gru2 = nn.GRU(
            input_size=hidden_size_gru1,
            hidden_size=hidden_size_gru2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.bn_final = nn.BatchNorm1d(hidden_size_gru2)
        self.fc1 = nn.Linear(hidden_size_gru2, hidden_size_gru2 // 2)
        self.fc2 = nn.Linear(hidden_size_gru2 // 2, output_size)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x = x.view(-1, features)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, features)
        
        gru1_out, _ = self.gru1(x)
        
        gru1_out = gru1_out.contiguous()
        batch_size, seq_len, hidden_size = gru1_out.size()
        gru1_out = gru1_out.view(-1, hidden_size)
        gru1_out = self.mid_bn(gru1_out)
        gru1_out = gru1_out.view(batch_size, seq_len, hidden_size)
        gru1_out = self.dropout(gru1_out)
        
        gru2_out, _ = self.gru2(gru1_out)
        
        final_out = gru2_out[:, -1, :]
        final_out = self.bn_final(final_out)
        final_out = self.dropout(final_out)
        final_out = torch.relu(self.fc1(final_out))
        final_out = self.fc2(final_out)
        
        return final_out

class LSTMLSTMModel(nn.Module):
    """LSTM-LSTM architecture for comparison"""
    def __init__(self, input_size, hidden_size_lstm1=64, hidden_size_lstm2=64, num_layers=2, output_size=1):
        super(LSTMLSTMModel, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm1,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.mid_bn = nn.BatchNorm1d(hidden_size_lstm1)
        self.dropout = nn.Dropout(0.2)
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_lstm1,
            hidden_size=hidden_size_lstm2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.bn_final = nn.BatchNorm1d(hidden_size_lstm2)
        self.fc1 = nn.Linear(hidden_size_lstm2, hidden_size_lstm2 // 2)
        self.fc2 = nn.Linear(hidden_size_lstm2 // 2, output_size)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x = x.view(-1, features)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, features)
        
        lstm1_out, _ = self.lstm1(x)
        
        lstm1_out = lstm1_out.contiguous()
        batch_size, seq_len, hidden_size = lstm1_out.size()
        lstm1_out = lstm1_out.view(-1, hidden_size)
        lstm1_out = self.mid_bn(lstm1_out)
        lstm1_out = lstm1_out.view(batch_size, seq_len, hidden_size)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        final_out = lstm2_out[:, -1, :]
        final_out = self.bn_final(final_out)
        final_out = self.dropout(final_out)
        final_out = torch.relu(self.fc1(final_out))
        final_out = self.fc2(final_out)
        
        return final_out

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
        'data_dir': "data_files/split_data",  # Directory containing ticker-specific files
        'seq_len': 15,
        'batch_size': 32,
        'epochs': 20,
        'hidden_size_lstm': 128,
        'hidden_size_gru': 128,
        'num_layers': 2,
        'ticker': None,
        'target_cols': ["bid", "ask"],
        'models_dir': "models",
        'performance_logs_dir': "performance_logs"
    }

def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logging.info("Starting Option Trading Model application")
    
    # Load configuration
    config = load_config()
    
    # Package model classes
    models = {
        'HybridRNNModel': HybridRNNModel,
        'GRUGRUModel': GRUGRUModel,
        'LSTMLSTMModel': LSTMLSTMModel
    }
    
    # Package handler functions
    handlers = {
        'handle_train_model': handle_train_model,
        'handle_run_model': handle_run_model,
        'handle_analyze_architecture': handle_analyze_architecture,
        'handle_benchmark_architectures': handle_benchmark_architectures
    }
    
    # Package data utilities
    data_utils = {
        'get_available_tickers': get_available_tickers,
        'select_ticker': select_ticker,
        'validate_paths': validate_paths,
        'StockOptionDataset': StockOptionDataset
    }
    
    # Package visualization utilities
    visualization_utils = {
        'save_and_display_results': save_and_display_results,
        'display_model_analysis': display_model_analysis
    }
    
    # Package performance utilities
    performance_utils = {
        'track_performance': track_performance,
        'benchmark_architectures': benchmark_architectures,
        'generate_architecture_comparison': generate_architecture_comparison,
        'visualize_architectures': visualize_architectures,
        'extended_train_model_with_tracking': extended_train_model_with_tracking
    }
    
    # Run application loop
    run_application_loop(
        config, 
        models, 
        handlers, 
        data_utils, 
        visualization_utils, 
        performance_utils
    )

if __name__ == "__main__":
    main()