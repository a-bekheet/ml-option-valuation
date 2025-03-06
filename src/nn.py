import torch
import torch.nn as nn
import logging
import os
import platform

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

def check_gpu():
    """
    Check if GPU acceleration is available and display information.
    Supports both CUDA (NVIDIA) and MPS (Apple Silicon) GPUs.
    
    Returns:
        tuple: (using_gpu, device_type) where device_type is 'cuda', 'mps', or 'cpu'
    """
    # Check for Apple Silicon Mac (M1/M2/M3)
    is_mac = platform.system() == 'Darwin'
    is_apple_silicon = is_mac and platform.processor() == 'arm'
    
    # Define variables with default values
    using_gpu = False
    device_type = 'cpu'
    
    # Check for MPS (Metal Performance Shaders) for Apple Silicon
    has_mps = False
    if is_apple_silicon:
        # Need to check if PyTorch was built with MPS support
        if hasattr(torch, 'has_mps') and torch.has_mps:
            has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        # For newer PyTorch versions
        elif hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
            has_mps = torch.backends.mps.is_available()
    
    # Check for CUDA (NVIDIA GPUs)
    has_cuda = torch.cuda.is_available()
    
    # Prioritize GPU based on what's available
    if has_cuda:
        using_gpu = True
        device_type = 'cuda'
        
        # Display CUDA GPU information
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        cuda_version = torch.version.cuda
        
        print(f"\n{'='*80}")
        print(f"NVIDIA GPU ACCELERATION AVAILABLE")
        print(f"{'='*80}")
        print(f"GPU Count: {gpu_count}")
        print(f"GPU Device: {gpu_name}")
        print(f"CUDA Version: {cuda_version}")
        
        # Print memory information if available
        try:
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            print(f"Memory Allocated: {memory_allocated / 1024**2:.2f} MB")
            print(f"Memory Reserved: {memory_reserved / 1024**2:.2f} MB")
        except:
            pass
        
    elif has_mps:
        using_gpu = True
        device_type = 'mps'
        
        # Display Apple Silicon GPU information
        print(f"\n{'='*80}")
        print(f"APPLE SILICON GPU ACCELERATION AVAILABLE")
        print(f"{'='*80}")
        print(f"Device: Apple {platform.processor().capitalize()} GPU")
        print(f"PyTorch MPS: {has_mps}")
        print(f"Operating System: {platform.system()} {platform.mac_ver()[0] if is_mac else ''}")
        
    else:
        # No GPU acceleration available
        print(f"\n{'='*80}")
        print(f"NO GPU ACCELERATION AVAILABLE - RUNNING ON CPU")
        print(f"{'='*80}")
        print("Training will be significantly slower on CPU.")
        if is_apple_silicon:
            print("Your Mac has Apple Silicon, but PyTorch MPS acceleration is not available.")
            print("Consider installing PyTorch with MPS support.")
        else:
            print("Consider running on a system with a GPU.")
        
    print(f"{'='*80}")
    return using_gpu, device_type

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
    # Check GPU availability and determine device type
    using_gpu, device_type = check_gpu()
    
    # Set default values
    config = {
        'data_dir': "data_files/split_data",  # Directory containing ticker-specific files
        'seq_len': 15,
        'batch_size': 32,
        'epochs': 20,
        'hidden_size_lstm': 64,
        'hidden_size_gru': 64,
        'num_layers': 2,
        'ticker': None,
        'target_cols': ["bid", "ask"],
        'models_dir': "models",
        'performance_logs_dir': "performance_logs",
        'device': device_type  # Set to 'cuda', 'mps', or 'cpu' based on availability
    }
    
    # Adjust batch size based on device
    if device_type == 'cuda':
        # For CUDA GPUs, adjust based on available memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory < 4:  # Less than 4GB
                config['batch_size'] = 16
                print(f"Low GPU memory ({gpu_memory:.1f}GB), reducing batch size to {config['batch_size']}")
            elif gpu_memory >= 8:  # 8GB or more
                config['batch_size'] = 64
                print(f"High GPU memory ({gpu_memory:.1f}GB), increasing batch size to {config['batch_size']}")
        except:
            pass
    elif device_type == 'mps':
        # For Apple Silicon, use a moderate batch size
        # Apple Silicon shares memory with the system, so we're more conservative
        config['batch_size'] = 48
        print(f"Using batch size {config['batch_size']} for Apple Silicon GPU")
    
    return config

def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logging.info("Starting Option Trading Model application")
    
    # Load configuration with appropriate device settings
    config = load_config()
    
    # Log device being used
    logging.info(f"Using device: {config['device']}")
    
    # Set PyTorch to benchmark mode for better performance (if using CUDA)
    if config['device'] == 'cuda':
        torch.backends.cudnn.benchmark = True
    
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