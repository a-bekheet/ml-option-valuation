import torch
import torch.nn as nn
import logging
import os
import platform
import sys
from pathlib import Path

from utils.data_utils import get_available_tickers, select_ticker, validate_paths, StockOptionDataset
from utils.model_utils import (
    train_model, analyze_model_architecture, load_model, 
    run_existing_model, calculate_errors, list_available_models, select_model,
    handle_train_model, handle_run_model, handle_analyze_architecture, handle_benchmark_architectures,
    run_existing_model_with_visualization, visualize_predictions
)
from utils.menu_utils import display_menu, run_application_loop
from utils.visualization_utils import save_and_display_results, display_model_analysis, plot_predictions
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
    is_mac = platform.system() == 'Darwin'
    is_apple_silicon = is_mac and platform.processor() == 'arm'

    using_gpu = False
    device_type = 'cpu'
    has_mps = False

    if is_apple_silicon:
        # Check for MPS (Metal Performance Shaders) for Apple Silicon
        # Updated check for newer PyTorch versions
        try:
            has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except AttributeError: # Fallback for older PyTorch versions
            has_mps = False
            logging.warning("Could not determine MPS availability (AttributeError). Assuming MPS is unavailable.")

    has_cuda = torch.cuda.is_available()

    if has_cuda:
        using_gpu = True
        device_type = 'cuda'
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        cuda_version = torch.version.cuda
        print(f"\n{'='*80}\nNVIDIA GPU ACCELERATION AVAILABLE\n{'='*80}")
        print(f"GPU Count: {gpu_count}\nGPU Device: {gpu_name}\nCUDA Version: {cuda_version}")
        try:
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"Memory Allocated: {memory_allocated:.2f} MB\nMemory Reserved: {memory_reserved:.2f} MB")
        except: pass # Ignore memory info errors
    elif has_mps:
        using_gpu = True
        device_type = 'mps'
        print(f"\n{'='*80}\nAPPLE SILICON GPU ACCELERATION AVAILABLE\n{'='*80}")
        print(f"Device: Apple {platform.processor().capitalize()} GPU")
        print(f"PyTorch MPS Available: {has_mps}")
        print(f"Operating System: {platform.system()} {platform.mac_ver()[0] if is_mac else ''}")
    else:
        print(f"\n{'='*80}\nNO GPU ACCELERATION AVAILABLE - RUNNING ON CPU\n{'='*80}")
        print("Training will be significantly slower on CPU.")
        if is_apple_silicon:
            print("Your Mac has Apple Silicon, but PyTorch MPS acceleration is not available or built.")
            print("Ensure you have the correct PyTorch version for MPS.")
        else:
            print("Consider running on a system with an NVIDIA GPU (CUDA) for faster training.")

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
    """Load configuration, detect environment, and set paths."""
    logging.info("Loading configuration...")
    # Assume check_gpu() is defined elsewhere and returns (using_gpu, device_type)
    # Placeholder if check_gpu is not available in this context:
    try:
        using_gpu, device_type = check_gpu()
    except NameError:
        logging.warning("check_gpu() function not found, assuming CPU.")
        using_gpu, device_type = False, 'cpu'


    # Environment detection
    IN_COLAB = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ
    if IN_COLAB:
        logging.info("Running in Google Colab environment.")
        print("It looks like you're running in Google Colab.")
        print("Please ensure your Google Drive is mounted if data/models are stored there.")
        print("Example mount command: from google.colab import drive; drive.mount('/content/drive')")
        # Define Colab base paths (assuming Drive is mounted at /content/drive)
        # MODIFY 'MyDrive/your_project_folder' TO YOUR ACTUAL PROJECT PATH IN DRIVE
        drive_base = Path('/content/drive/MyDrive/option-ml-prediction') #<- ADJUST THIS
        if not drive_base.exists():
             logging.warning(f"Google Drive path '{drive_base}' not found. Using /content/ as base.")
             # Fallback to /content if drive path incorrect or not mounted
             base_path = Path('/content')
             colab_data_dir = base_path / 'data_files/split_data'
             colab_models_dir = base_path / 'models'
             colab_perf_logs_dir = base_path / 'performance_logs'
             colab_viz_dir = base_path / 'plots'
        else:
             base_path = drive_base
             colab_data_dir = base_path / 'data_files/split_data'
             colab_models_dir = base_path / 'models'
             colab_perf_logs_dir = base_path / 'performance_logs'
             colab_viz_dir = base_path / 'plots' # Separate plots directory

        # Ensure Colab directories exist
        colab_data_dir.mkdir(parents=True, exist_ok=True)
        colab_models_dir.mkdir(parents=True, exist_ok=True)
        colab_perf_logs_dir.mkdir(parents=True, exist_ok=True)
        colab_viz_dir.mkdir(parents=True, exist_ok=True)

        default_data_dir = str(colab_data_dir)
        default_models_dir = str(colab_models_dir)
        default_perf_logs_dir = str(colab_perf_logs_dir)
        default_viz_dir = str(colab_viz_dir)
    else:
        logging.info("Running in local environment.")
        # --- MODIFICATION START: Use Absolute Paths ---
        # Get the directory where this script (nn.py) resides
        # Note: This requires Python 3.9+ for Path(__file__) behavior in scripts
        # If using older Python, alternative methods might be needed.
        try:
            script_dir = Path(__file__).parent.resolve()
            # Assume nn.py is in src/, so project root is one level up
            project_root = script_dir.parent
        except NameError:
             # Fallback if __file__ is not defined (e.g., interactive session)
             # This will use the current working directory as the project root.
             logging.warning("__file__ not defined. Using current working directory as project root. Ensure you run from the project root.")
             project_root = Path('.').resolve()

        # Construct absolute paths relative to the project root
        default_data_dir = str(project_root / "data_files/split_data")
        default_models_dir = str(project_root / "models")
        default_perf_logs_dir = str(project_root / "performance_logs")
        default_viz_dir = str(project_root / "plots") # Separate plots directory
        # --- MODIFICATION END ---

    # Default model/training parameters
    config = {
        'seq_len': 30,
        'batch_size': 32, # Default, will be adjusted based on device
        'epochs': 20,
        'lr': 1e-3, # Added Learning Rate
        'hidden_size_lstm': 64,
        'hidden_size_gru': 64,
        'num_layers': 2,
        'ticker': None, # Will be set later by user or default
        'target_cols': ["bid", "ask"],
        'device': device_type,
        # --- Paths defined based on environment ---
        'data_dir': default_data_dir,
        'models_dir': default_models_dir,
        'performance_logs_dir': default_perf_logs_dir,
        'viz_dir': default_viz_dir # Visualization output directory
    }

    # Adjust batch size based on device (existing logic)
    if device_type == 'cuda':
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory < 4: config['batch_size'] = 16
            elif gpu_memory >= 12: config['batch_size'] = 64 # Increased threshold for larger batch
            else: config['batch_size'] = 32
            logging.info(f"CUDA GPU memory: {gpu_memory:.1f}GB. Setting batch size to {config['batch_size']}")
        except Exception as e:
            logging.warning(f"Could not get GPU memory, using default batch size {config['batch_size']}. Error: {e}")
    elif device_type == 'mps':
        config['batch_size'] = 48 # Keep moderate for MPS
        logging.info(f"Apple Silicon MPS detected. Setting batch size to {config['batch_size']}")
    else: # CPU
         logging.info(f"Running on CPU. Using default batch size {config['batch_size']}")

    logging.info("Configuration loaded:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")

    return config

def main():
    """Main application entry point."""
    setup_logging() # Setup logging first
    logging.info("Starting Option Trading Model application")

    config = load_config()
    logging.info(f"Using device: {config['device']}")
    if config['device'] == 'cuda':
        torch.backends.cudnn.benchmark = True
        logging.info("Enabled CuDNN benchmark mode.")

    models = { 'HybridRNNModel': HybridRNNModel, 'GRUGRUModel': GRUGRUModel, 'LSTMLSTMModel': LSTMLSTMModel }
    handlers = { 'handle_train_model': handle_train_model, 'handle_run_model': handle_run_model, 'handle_analyze_architecture': handle_analyze_architecture, 'handle_benchmark_architectures': handle_benchmark_architectures }
    data_utils = { 'get_available_tickers': get_available_tickers, 'select_ticker': select_ticker, 'validate_paths': validate_paths, 'StockOptionDataset': StockOptionDataset }
    visualization_utils = { 'save_and_display_results': save_and_display_results, 'display_model_analysis': display_model_analysis, 'plot_predictions': plot_predictions }
    performance_utils = { 'track_performance': track_performance, 'benchmark_architectures': benchmark_architectures, 'generate_architecture_comparison': generate_architecture_comparison, 'visualize_architectures': visualize_architectures, 'extended_train_model_with_tracking': extended_train_model_with_tracking }

    run_application_loop( config, models, handlers, data_utils, visualization_utils, performance_utils )

if __name__ == "__main__":
    main()