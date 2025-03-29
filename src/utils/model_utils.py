import json
import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from utils.visualization_utils import plot_predictions


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_errors(y_true, y_pred):
    """Calculate various error metrics between predicted and actual values."""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Calculate errors (averaged over both targets)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def analyze_model_architecture(model, input_size=None, seq_len=15, batch_size=32):
    """
    Analyze the architecture of the model, including parameter count and tensor shapes.
    Ensures dummy input is on the correct device.

    Args:
        model: The PyTorch model instance.
        input_size (int): The number of input features. If None, tries to infer from model.
        seq_len (int): Sequence length for dummy input.
        batch_size (int): Batch size for dummy input.

    Returns:
        dict: Dictionary containing architecture analysis results.
    """
    if input_size is None:
        # Try to infer input size (e.g., from the first linear layer or LSTM/GRU input_size)
        # This is a basic inference, might need adjustment based on your specific models
        try:
            if hasattr(model, 'lstm') and hasattr(model.lstm, 'input_size'):
                input_size = model.lstm.input_size
            elif hasattr(model, 'gru1') and hasattr(model.gru1, 'input_size'):
                input_size = model.gru1.input_size
            elif hasattr(model, 'lstm1') and hasattr(model.lstm1, 'input_size'):
                input_size = model.lstm1.input_size
            # Add fallbacks for other potential first layers if necessary
            else:
                 # Attempt to find any LSTM or GRU layer
                 for layer in model.children():
                      if isinstance(layer, (torch.nn.LSTM, torch.nn.GRU)):
                           input_size = layer.input_size
                           break
                 if input_size is None:
                      raise ValueError("Could not automatically infer input_size.")
            logging.info(f"Inferred input_size for analysis: {input_size}")
        except Exception as e:
            logging.error(f"Failed to infer input_size for architecture analysis: {e}")
            logging.error("Please provide input_size explicitly to analyze_model_architecture.")
            # Return basic parameter counts if size inference fails
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'layer_shapes': {"Error": "Could not perform forward pass for shape analysis."}
            }


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Determine the device the model is on
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu") # Model has no parameters? Fallback to CPU
    logging.info(f"Analyzing architecture on device: {device}")

    # Create dummy input and move it to the model's device
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    dummy_input = dummy_input.to(device) # <-- Explicitly move to device

    layer_shapes = {}

    def hook_fn(module, input_tensors, output_tensors, name):
        def get_tensor_shape(t):
            if isinstance(t, torch.Tensor):
                return tuple(t.shape)
            elif isinstance(t, (list, tuple)):
                # Handle nested tuples/lists, e.g., LSTM output (output, (h_n, c_n))
                return tuple(get_tensor_shape(sub_t) for sub_t in t if isinstance(sub_t, (torch.Tensor, list, tuple)))
            return None

        # input_tensors is typically a tuple
        input_shapes = [get_tensor_shape(t) for t in input_tensors]
        # If input_shapes is like [(shape1,), (shape2,)], flatten it
        if len(input_shapes) == 1:
             input_shapes = input_shapes[0]

        layer_shapes[name] = {
            'input_shape': input_shapes,
            'output_shape': get_tensor_shape(output_tensors)
        }

    hooks = []
    for name, layer in model.named_children():
        # Use lambda with default argument capture for name
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, layer_name=name: hook_fn(m, i, o, layer_name)
        ))

    # Perform forward pass with dummy input
    model.eval() # Ensure model is in eval mode for analysis
    try:
        with torch.no_grad():
            logging.debug(f"Performing forward pass for analysis with dummy input shape: {dummy_input.shape} on device {dummy_input.device}")
            _ = model(dummy_input) # <--- Error occurred around here previously
        logging.debug("Forward pass for analysis successful.")
    except Exception as e:
         logging.error(f"Error during dummy forward pass in analyze_model_architecture: {e}")
         logging.error("Layer shape analysis might be incomplete.")
         layer_shapes = {"Error": f"Forward pass failed: {e}"} # Indicate analysis failure
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layer_shapes': layer_shapes
    }

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss() # type: ignore
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    def log_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    early_stopping = EarlyStopping(patience=5)
    
    model.to(device)
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]',
                         leave=False, unit='batch')
        
        for x_seq, y_val in train_pbar:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_val)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            
            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0.0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]',
                       leave=False, unit='batch')
        
        with torch.no_grad():
            for x_seq, y_val in val_pbar:
                x_seq, y_val = x_seq.to(device), y_val.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_val)
                total_val_loss += loss.item()
                
                # Update progress bar with current loss
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        current_lr = log_lr(optimizer)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break
    
    return train_losses, val_losses

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

def get_model_class_from_name(model_name, HybridRNNModel, GRUGRUModel, LSTMLSTMModel):
    """Determine the model class based on the model name."""
    try:
        logging.info(f"Determining model class from name: {model_name}")
        model_name = model_name.lower()
        
        if "gru-gru" in model_name:
            logging.info("Selected GRU-GRU model architecture")
            return GRUGRUModel, "GRU-GRU"
        elif "lstm-lstm" in model_name:
            logging.info("Selected LSTM-LSTM model architecture")
            return LSTMLSTMModel, "LSTM-LSTM"
        else:
            logging.info("Selected LSTM-GRU hybrid model architecture (default)")
            return HybridRNNModel, "LSTM-GRU"
            
    except Exception as e:
        logging.error(f"Error in get_model_class_from_name: {str(e)}")
        # Default to HybridRNNModel if there's an error
        return HybridRNNModel, "LSTM-GRU"
        
def load_model(model_path, model_class, input_size, 
               hidden_size_lstm=64, hidden_size_gru=64,  # Change from 128 to 64
               hidden_size_lstm1=64, hidden_size_lstm2=64,  # Change from 128 to 64
               hidden_size_gru1=64, hidden_size_gru2=64,  # Change from 128 to 64
               num_layers=2, output_size=1):
    """
    Load a saved model with proper architecture class.
    Supports all three architecture types.
    """
    try:
        logging.info(f"Loading model from: {model_path}")
        logging.info(f"Model class: {model_class.__name__}")
        logging.info(f"Input size: {input_size}, Output size: {output_size}")
        
        # Create the model instance
        if model_class.__name__ == 'HybridRNNModel':
            logging.info(f"Creating HybridRNNModel with hidden sizes: LSTM={hidden_size_lstm}, GRU={hidden_size_gru}")
            model = model_class(
                input_size=input_size,
                hidden_size_lstm=hidden_size_lstm,
                hidden_size_gru=hidden_size_gru,
                num_layers=num_layers,
                output_size=output_size
            )
        elif model_class.__name__ == 'GRUGRUModel':
            logging.info(f"Creating GRUGRUModel with hidden sizes: GRU1={hidden_size_gru1}, GRU2={hidden_size_gru2}")
            model = model_class(
                input_size=input_size,
                hidden_size_gru1=hidden_size_gru1,
                hidden_size_gru2=hidden_size_gru2,
                num_layers=num_layers,
                output_size=output_size
            )
        elif model_class.__name__ == 'LSTMLSTMModel':
            logging.info(f"Creating LSTMLSTMModel with hidden sizes: LSTM1={hidden_size_lstm1}, LSTM2={hidden_size_lstm2}")
            model = model_class(
                input_size=input_size,
                hidden_size_lstm1=hidden_size_lstm1,
                hidden_size_lstm2=hidden_size_lstm2,
                num_layers=num_layers,
                output_size=output_size
            )
        else:
            raise ValueError(f"Unknown model class: {model_class.__name__}")
            
        # Load the model weights
        logging.info("Loading model state dict")
        model.load_state_dict(torch.load(model_path))
        logging.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def run_existing_model(model_path, model_class, dataset, target_cols=["bid", "ask"]):
    """Load and run predictions with an existing model."""
    try:
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logging.info(f"Running model for {dataset.ticker} with {len(dataset)} samples")
        logging.info(f"Dataset features: {dataset.n_features}, targets: {len(dataset.target_cols)}")
        
        # Load the model
        model = load_model(
            model_path, 
            model_class,
            input_size=dataset.n_features, 
            output_size=len(dataset.target_cols)
        )
        
        if model is None:
            raise ValueError("Model could not be loaded")
            
        model.eval()
        
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        all_y_true = []
        all_y_pred = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        print(f"\nMaking predictions for {dataset.ticker}...")
        with torch.no_grad():
            for x_seq, y_val in data_loader:
                x_seq, y_val = x_seq.to(device), y_val.to(device)
                y_pred = model(x_seq)
                all_y_true.extend(y_val.cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy())
        
        errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
        print("\nPrediction Metrics:")
        print("-" * 50)
        print(f"MSE: {errors['mse']:.6f}")
        print(f"RMSE: {errors['rmse']:.6f}")
        print(f"MAE: {errors['mae']:.6f}")
        print(f"MAPE: {errors['mape']:.2f}%")
        
        # Import visualization_utils here to avoid circular imports
        from utils.visualization_utils import plot_predictions
        
        # Create plots for predicted vs actual values
        models_dir = os.path.dirname(model_path)
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        plot_predictions(all_y_true, all_y_pred, target_cols, dataset.ticker, models_dir, timestamp)
        
        return all_y_true, all_y_pred
        
    except Exception as e:
        logging.error(f"Error in run_existing_model: {str(e)}")
        raise

def load_scaling_params(ticker: str, data_dir: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load the scaling parameters (mean, scale/std) for a specific ticker.
    Looks for the file in the 'by_ticker/scaling_params' subdirectory.

    Args:
        ticker (str): The ticker symbol.
        data_dir (str): The base data directory (e.g., 'data_files/split_data').

    Returns:
        Optional[Dict[str, Dict[str, float]]]: Dictionary of scaling parameters or None if not found.
    """
    params_path = os.path.join(data_dir, 'by_ticker', 'scaling_params', f"{ticker}_scaling_params.json")
    logging.info(f"Attempting to load scaling parameters from: {params_path}")
    if not os.path.exists(params_path):
        logging.error(f"Scaling parameters file not found: {params_path}")
        return None
    try:
        with open(params_path, 'r') as f:
            scaling_params = json.load(f)
        logging.info(f"Successfully loaded scaling parameters for {ticker}.")
        return scaling_params
    except Exception as e:
        logging.error(f"Error loading or parsing scaling parameters file {params_path}: {e}")
        return None

def recover_original_values(
    normalized_values: np.ndarray,
    column_names: List[str],
    scaling_params: Dict[str, Dict[str, float]]
) -> np.ndarray:
    """
    Recover original values from normalized values using scaling parameters.

    Args:
        normalized_values (np.ndarray): Array of normalized values (samples x features).
        column_names (List[str]): List of column names corresponding to the feature dimension.
        scaling_params (Dict[str, Dict[str, float]]): Dictionary of scaling parameters ({'mean': m, 'scale': s}).

    Returns:
        np.ndarray: Array with recovered original values.
    """
    if not scaling_params:
         logging.warning("Received empty scaling parameters. Cannot un-normalize data.")
         return normalized_values # Return original if params are missing

    original_values = normalized_values.copy()
    num_cols = normalized_values.shape[1]

    if len(column_names) != num_cols:
        logging.error(f"Mismatch between number of column names ({len(column_names)}) and data columns ({num_cols}). Cannot reliably un-normalize.")
        return normalized_values # Return original on mismatch

    logging.info(f"Recovering original values for columns: {column_names}")
    for i, col_name in enumerate(column_names):
        if col_name in scaling_params:
            params = scaling_params[col_name]
            mean = params.get('mean', 0)
            # Use 'scale' if present (from StandardScaler), otherwise fallback to 'std'
            scale = params.get('scale', params.get('std', 1))
            if scale == 0: # Avoid division by zero if scale is zero
                 logging.warning(f"Scaling factor is zero for column '{col_name}'. Skipping un-normalization for this column.")
                 continue
            # Apply inverse transform: X_orig = X_scaled * scale + mean
            original_values[:, i] = normalized_values[:, i] * scale + mean
            logging.debug(f"Un-normalized '{col_name}' using mean={mean}, scale={scale}")
        else:
            logging.warning(f"No scaling parameters found for column '{col_name}'. Leaving it as is.")

    return original_values

def visualize_predictions(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         target_cols: List[str],
                         ticker: str,
                         scaling_params: Optional[Dict[str, Dict[str, float]]] = None,
                         output_dir: str = "prediction_visualizations",
                         show_plot: bool = True,
                         n_samples: int = 100) -> Dict[str, str]:
    """
    Visualize model predictions against actual values, optionally recovering original values.
    
    Args:
        y_true: Ground truth values (normalized)
        y_pred: Predicted values (normalized)
        target_cols: Names of target columns
        ticker: Ticker symbol
        scaling_params: Optional scaling parameters for recovery to original values
        output_dir: Directory to save visualizations
        show_plot: Whether to display the plots
        n_samples: Number of samples to visualize
        
    Returns:
        Dictionary mapping target columns to saved plot file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.cpu().numpy()
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.cpu().numpy()
    
    # Ensure arrays are 2D
    if len(y_true_np.shape) == 1:
        y_true_np = y_true_np.reshape(-1, 1)
    if len(y_pred_np.shape) == 1:
        y_pred_np = y_pred_np.reshape(-1, 1)
    
    # Limit to the specified number of samples
    if n_samples < y_true_np.shape[0]:
        y_true_np = y_true_np[:n_samples]
        y_pred_np = y_pred_np[:n_samples]
    
    # Recover original values if scaling parameters are provided
    if scaling_params:
        y_true_original = recover_original_values(y_true_np, target_cols, scaling_params)
        y_pred_original = recover_original_values(y_pred_np, target_cols, scaling_params)
        
        # Add debugging for recovery process
        print("\n=== RECOVERY PROCESS DEBUG INFO ===")
        print(f"Recovery for {y_true_np.shape[0]} samples and {len(target_cols)} target columns")
        
        # Print a sample of before/after recovery
        sample_size = min(3, y_true_np.shape[0])
        for i in range(sample_size):
            print(f"\nSample {i+1}:")
            for j, col in enumerate(target_cols):
                norm_true = y_true_np[i, j] if len(y_true_np.shape) > 1 else y_true_np[i]
                norm_pred = y_pred_np[i, j] if len(y_pred_np.shape) > 1 else y_pred_np[i]
                
                orig_true = y_true_original[i, j] if len(y_true_original.shape) > 1 else y_true_original[i]
                orig_pred = y_pred_original[i, j] if len(y_pred_original.shape) > 1 else y_pred_original[i]
                
                print(f"  {col}: normalized [true={norm_true:.6f}, pred={norm_pred:.6f}] → "
                      f"original [true={orig_true:.6f}, pred={orig_pred:.6f}]")
                
                # Double-check the recovery calculation
                if col in scaling_params:
                    mean = scaling_params[col].get('mean', 0)
                    scale = scaling_params[col].get('scale', 1)
                    manual_true = norm_true * scale + mean
                    manual_pred = norm_pred * scale + mean
                    
                    # Check if our manual calculation matches the function's output
                    true_match = np.isclose(manual_true, orig_true)
                    pred_match = np.isclose(manual_pred, orig_pred)
                    
                    if not true_match or not pred_match:
                        print(f"    WARNING: Recovery mismatch!")
                        print(f"    Manual calculation: true={manual_true:.6f}, pred={manual_pred:.6f}")
                        print(f"    Function output:   true={orig_true:.6f}, pred={orig_pred:.6f}")
                        print(f"    Scaling parameters: mean={mean}, scale={scale}")
    else:
        y_true_original = y_true_np
        y_pred_original = y_pred_np
    

    # Create visualizations for each target column
    plot_files = {}
    
    for i, col in enumerate(target_cols):
        # Extract the values for this column
        true_values = y_true_original[:, i]
        pred_values = y_pred_original[:, i]
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create the main plot with actual and predicted values
        plt.subplot(2, 1, 1)
        plt.plot(true_values, label=f'Actual {col}', linewidth=2)
        plt.plot(pred_values, label=f'Predicted {col}', linewidth=2, linestyle='--')
        plt.title(f'{ticker} - {col} Predictions vs Actual Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add prediction error plot
        plt.subplot(2, 1, 2)
        errors = pred_values - true_values
        plt.bar(range(len(errors)), errors, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f'Prediction Error ({col})')
        plt.xlabel('Sample Index')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(output_dir, f"{ticker}_{col}_predictions_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files[col] = plot_file
        
        if not show_plot:
            plt.close()
    
    # Create a summary visualization with all targets
    if len(target_cols) > 1:
        plt.figure(figsize=(14, 10))
        
        for i, col in enumerate(target_cols):
            plt.subplot(len(target_cols), 1, i+1)
            true_values = y_true_original[:, i]
            pred_values = y_pred_original[:, i]
            
            plt.plot(true_values, label=f'Actual {col}', linewidth=2)
            plt.plot(pred_values, label=f'Predicted {col}', linewidth=2, linestyle='--')
            plt.title(f'{col}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{ticker} - Model Predictions Summary', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to accommodate suptitle
        
        # Save summary plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(output_dir, f"{ticker}_predictions_summary_{timestamp}.png")
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plot_files['summary'] = summary_file
        
        if not show_plot:
            plt.close()
    
    return plot_files

def run_existing_model_with_visualization(
    model_path: str,
    model_class: Any,
    dataset: Any, # Should be an instance of StockOptionDataset
    target_cols: List[str] = ["bid", "ask"],
    visualize: bool = True,
    n_samples_plot: int = 250, # Renamed from n_samples to avoid confusion
    data_dir: Optional[str] = None, # Needed to load scaling params
    output_dir: str = "prediction_visualizations" # Changed default
) -> Tuple[Dict[str, float], Optional[Dict[str, str]]]:
    """
    Load and run predictions, un-normalize results, calculate original-scale metrics,
    and optionally visualize with annotations.

    Args:
        model_path (str): Path to the saved model.
        model_class (Any): Model class (e.g., HybridRNNModel).
        dataset (Any): Instance of StockOptionDataset for the desired ticker.
        target_cols (List[str]): Target columns predicted by the model.
        visualize (bool): Whether to generate plots.
        n_samples_plot (int): Number of samples for time series plots.
        data_dir (Optional[str]): Base data directory (e.g., 'data_files/split_data') needed for scaling params.
        output_dir (str): Directory to save visualizations.

    Returns:
        Tuple[Dict[str, float], Optional[Dict[str, str]]]:
            - Dictionary of error metrics (calculated on *original* scale).
            - Dictionary mapping plot types to saved file paths if visualized, else None.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Load Model and Get Predictions (Normalized) ---
    try:
        # Assume dataset provides necessary info like n_features, n_targets, ticker
        model = load_model(
            model_path, model_class,
            input_size=dataset.n_features, output_size=dataset.n_targets
        )
        if model is None: raise ValueError("Failed to load model.")
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        print(f"\nError loading model: {e}")
        return {}, None # Return empty results on failure

    # Determine device
    try:
         device = next(model.parameters()).device
    except StopIteration:
         device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logging.info(f"Running model on device: {device}")

    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_y_true_norm = []
    all_y_pred_norm = []

    print(f"\nMaking predictions for {dataset.ticker}...")
    with torch.no_grad():
        for x_seq, y_val in data_loader:
            x_seq = x_seq.to(device)
            # y_val stays on CPU for easier numpy conversion later
            y_pred = model(x_seq)
            all_y_true_norm.extend(y_val.cpu().numpy())
            all_y_pred_norm.extend(y_pred.cpu().numpy())

    y_true_norm_np = np.array(all_y_true_norm)
    y_pred_norm_np = np.array(all_y_pred_norm)

    # --- Un-normalize Predictions and Actuals ---
    y_true_orig = y_true_norm_np # Default if un-normalization fails
    y_pred_orig = y_pred_norm_np
    scaling_params = None
    un_normalized = False

    if data_dir:
        scaling_params = load_scaling_params(dataset.ticker, data_dir)
        if scaling_params:
            # Ensure target_cols matches the order/names expected by recover_original_values
            try:
                # We need to un-normalize based on the target column names
                y_true_orig = recover_original_values(y_true_norm_np, target_cols, scaling_params)
                y_pred_orig = recover_original_values(y_pred_norm_np, target_cols, scaling_params)
                print(f"\nSuccessfully un-normalized predictions and actuals for {dataset.ticker}.")
                un_normalized = True
            except Exception as e:
                logging.error(f"Failed to un-normalize data for {dataset.ticker}: {e}")
                print("\nWarning: Could not un-normalize data. Metrics and plots will use normalized values.")
        else:
            print("\nWarning: Scaling parameters not found. Metrics and plots will use normalized values.")
    else:
        print("\nWarning: data_dir not provided. Cannot load scaling parameters. Metrics and plots will use normalized values.")

    # --- Calculate Metrics (Original Scale if possible) ---
    print("\nCalculating Prediction Metrics" + (" (Original Scale)" if un_normalized else " (Normalized Scale)") + ":")
    # Use original scale arrays if available, otherwise use normalized ones
    y_true_calc = torch.tensor(y_true_orig)
    y_pred_calc = torch.tensor(y_pred_orig)
    errors_orig = calculate_errors(y_true_calc, y_pred_calc) # Assumes calculate_errors handles numpy/torch tensor

    # Format metrics for printing/plotting
    rmse_val = errors_orig.get('rmse', np.nan)
    mae_val = errors_orig.get('mae', np.nan)

    print("-" * 50)
    print(f"RMSE: {rmse_val:.4f}") # Use more decimal places for dollar values potentially
    print(f"MAE:  {mae_val:.4f}")
    # Optionally calculate and print other metrics like MAPE, Dir Acc if needed,
    # but calculate them on the *original scale* data (y_true_orig, y_pred_orig)
    # MAPE might still be unstable if original prices are near zero.

    # --- Visualize Results (If Requested) ---
    viz_files = None
    if visualize:
        print("\nGenerating visualizations...")
        try:
            # Pass the ORIGINAL SCALE data and metrics to the plotting function
            viz_files = plot_predictions(
                y_true=y_true_orig, # Pass un-normalized
                y_pred=y_pred_orig, # Pass un-normalized
                target_cols=target_cols,
                ticker=dataset.ticker,
                output_dir=output_dir,
                # Pass metrics for annotation
                rmse=rmse_val,
                mae=mae_val,
                n_samples_plot=n_samples_plot # Pass plotting sample limit
            )
            print("\nVisualizations saved to:")
            if viz_files:
                 for target, file_path in viz_files.items():
                      print(f"  {target}: {file_path}")
            else:
                 print("  Plotting function did not return file paths.")
        except Exception as e:
            logging.error(f"Error during visualization: {e}")
            print(f"\nError generating plots: {e}")

    return errors_orig, viz_files # Return original scale errors
# Handler Functions moved from nn.py
def handle_train_model(config, HybridRNNModel, GRUGRUModel, LSTMLSTMModel,
                       save_and_display_results, extended_train_model_with_tracking,
                       get_available_tickers, select_ticker, StockOptionDataset):
    """Handle the model training workflow with optional Greeks/Rolling Features and enhanced logging."""
    try:
        logging.info("Starting model training workflow...")

        # --- User Prompts ---
        use_tracking = input("\nUse detailed performance tracking log? (y/n): ").lower().startswith('y')
        logging.info(f"Performance tracking enabled: {use_tracking}")

        # Prompt for Greeks
        use_greeks_input = input("Include Option Greeks features? (y/n): ").lower()
        include_greeks_flag = use_greeks_input == 'y'
        logging.info(f"Include Option Greeks features: {include_greeks_flag}")

        # >> NEW: Prompt for Rolling Window Features <<
        use_rolling_input = input("Include Rolling Window features? (y/n): ").lower()
        include_rolling_flag = use_rolling_input == 'y'
        logging.info(f"Include Rolling Window features: {include_rolling_flag}")

        # Prompt for architecture
        print("\nSelect model architecture:")
        print("1. LSTM-GRU Hybrid (default)")
        print("2. GRU-GRU")
        print("3. LSTM-LSTM")
        arch_choice = input("Enter choice (1-3): ").strip()
        # [ ... same architecture selection logic ... ]
        if arch_choice == "2": architecture_type = "GRU-GRU"; SelectedModelClass = GRUGRUModel
        elif arch_choice == "3": architecture_type = "LSTM-LSTM"; SelectedModelClass = LSTMLSTMModel
        else: architecture_type = "LSTM-GRU"; SelectedModelClass = HybridRNNModel
        logging.info(f"Selected architecture: {architecture_type}")

        # --- Data Preparation ---
        tickers, counts = get_available_tickers(config['data_dir'])
        if not tickers: raise ValueError("No tickers available in the data directory.") # Raise error
        ticker = select_ticker(tickers, counts)
        if not ticker: logging.warning("No ticker selected."); return
        logging.info(f"Selected ticker: {ticker}")

        # Initialize dataset with *both* flags
        logging.info(f"Initializing dataset for {ticker}, include_greeks={include_greeks_flag}, include_rolling={include_rolling_flag}...")
        try:
             dataset = StockOptionDataset(
                 data_dir=config['data_dir'], ticker=ticker,
                 seq_len=config['seq_len'], target_cols=config['target_cols'],
                 include_greeks=include_greeks_flag, # Pass Greeks flag
                 include_rolling_features=include_rolling_flag, # Pass Rolling flag
                 verbose=True
             )
        # [ ... same dataset error handling ... ]
        except FileNotFoundError as fnf_err: logging.error(f"Dataset file not found: {fnf_err}"); print(f"\nError: Data file not found for {ticker}."); return
        except Exception as data_err: logging.error(f"Error initializing dataset: {data_err}"); print(f"\nError creating dataset: {data_err}"); return


        # Get features actually used (this will now reflect both flags)
        features_used_in_run = dataset.feature_cols
        logging.info(f"Number of features used in this run: {dataset.n_features}")
        logging.debug(f"Features list: {features_used_in_run}")

        # [ ... same data length checks ... ]
        if dataset.n_features == 0: raise ValueError("Dataset loaded with 0 features.")
        seq_len = config.get('seq_len', 15)
        if len(dataset) < seq_len + 1: raise ValueError(f"Insufficient samples for {ticker} ({len(dataset)}) for sequence length {seq_len}.")

        # [ ... same Data Splitting & Loaders ... ]
        total_len = len(dataset); train_len = int(0.80 * total_len); val_len = int(0.10 * total_len)
        if train_len==0 or val_len==0 or (total_len-train_len-val_len)==0: raise ValueError(f"Dataset {ticker} too small for split.")
        indices = list(range(total_len)); train_ds = Subset(dataset, indices[:train_len])
        val_ds = Subset(dataset, indices[train_len:train_len+val_len]); test_ds = Subset(dataset, indices[train_len+val_len:])
        batch_size = config.get('batch_size', 32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
        logging.info(f"DataLoaders created: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} samples.")

        # --- Model Initialization (No changes needed here, uses dataset.n_features) ---
        device = config.get('device', 'cpu')
        model_input_size = dataset.n_features # Automatically adjusts based on dataset flags
        model_output_size = dataset.n_targets
        num_layers_config = config.get('num_layers', 2)
        logging.info(f"Initializing model {architecture_type} with input_size={model_input_size}, output_size={model_output_size}")
        # [ ... same model initialization logic using if/elif/else ... ]
        if SelectedModelClass == HybridRNNModel: model = SelectedModelClass(input_size=model_input_size, hidden_size_lstm=config.get('hidden_size_lstm', 64), hidden_size_gru=config.get('hidden_size_gru', 64), num_layers=num_layers_config, output_size=model_output_size)
        elif SelectedModelClass == GRUGRUModel: model = SelectedModelClass(input_size=model_input_size, hidden_size_gru1=config.get('hidden_size_gru', 64), hidden_size_gru2=config.get('hidden_size_gru', 64), num_layers=num_layers_config, output_size=model_output_size)
        elif SelectedModelClass == LSTMLSTMModel: model = SelectedModelClass(input_size=model_input_size, hidden_size_lstm1=config.get('hidden_size_lstm', 64), hidden_size_lstm2=config.get('hidden_size_lstm', 64), num_layers=num_layers_config, output_size=model_output_size)
        else: raise ValueError(f"Unknown model class: {SelectedModelClass.__name__}")
        print(f"\nInitialized {architecture_type} architecture on device: {device}")
        model.to(device)

        # --- Architecture Analysis (No changes needed here) ---
        logging.info("Analyzing model architecture...")
        model_analysis = analyze_model_architecture( model, input_size=model_input_size, seq_len=config['seq_len'] )
        logging.info(f"Model analysis complete: Total Params={model_analysis.get('total_parameters', 'N/A')}")

        # --- Training (No changes needed here, passes features_used_in_run) ---
        history = None; log_path = None
        if use_tracking:
            logging.info("Starting training with performance tracking...")
            # extended_train_model_with_tracking call remains the same
            model, history, log_path = extended_train_model_with_tracking(
                model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                epochs=config['epochs'], lr=config.get('lr', 1e-3), device=device, ticker=ticker,
                architecture_name=architecture_type, target_cols=config['target_cols'],
                used_features=features_used_in_run, # This list now depends on both flags
                model_analysis_dict=model_analysis
            )
            if log_path: print(f"\nPerformance log saved to: {log_path}")
        else:
            # Standard training logic remains the same
            logging.info("Starting standard training (no detailed log file)...")
            # [ ... standard training and evaluation logic ... ]
            # Ensure train_model, calculate_errors are imported/available
            train_losses, val_losses = train_model(model=model, train_loader=train_loader, val_loader=val_loader, epochs=config['epochs'], lr=config.get('lr', 1e-3), device=device)
            history = {'train_losses': train_losses, 'val_losses': val_losses}
            logging.info("Standard training complete. Evaluating on test set...")
            model.eval(); all_y_true = []; all_y_pred = []
            criterion = torch.nn.MSELoss()
            with torch.no_grad():
                 for x_seq, y_val in test_loader:
                      x_seq, y_val = x_seq.to(device), y_val.to(device)
                      y_pred = model(x_seq)
                      all_y_true.extend(y_val.cpu().numpy())
                      all_y_pred.extend(y_pred.cpu().numpy())
            errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
            print("\nFinal Test Set Metrics:"); print("-" * 50)
            print(f"MSE: {errors.get('mse', 'N/A'):.6f}"); print(f"RMSE: {errors.get('rmse', 'N/A'):.6f}")
            print(f"MAE: {errors.get('mae', 'N/A'):.6f}"); print(f"MAPE: {errors.get('mape', 'N/A'):.2f}%")
            history['y_true'] = all_y_true; history['y_pred'] = all_y_pred

        # --- Save Results (No changes needed here) ---
        if history:
            logging.info("Saving model and results...")
            # Ensure save_and_display_results is imported
            save_and_display_results( model=model, history=history, analysis=model_analysis,
                ticker=ticker, target_cols=config['target_cols'], models_dir=config['models_dir'] )
            logging.info("Model training workflow completed successfully.")
        else:
             logging.warning("Training did not produce history results. Skipping save.")

    # [ ... same error handling ... ]
    except FileNotFoundError as e: logging.error(f"File not found: {e}"); print(f"\nError: File not found: {e}")
    except ValueError as e: logging.error(f"Value error: {e}"); print(f"\nError: {e}")
    except Exception as e: import traceback; logging.exception(f"Unexpected error: {e}"); print(f"\nUnexpected error: {e}")

def handle_run_model(config, models_dir, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, get_available_tickers, select_ticker, StockOptionDataset):
    """Handle the model prediction workflow."""
    try:
        model_files = list_available_models(models_dir)
        if not model_files:
            return

        selected_model = select_model(model_files)
        if not selected_model:
            return

        model_path = os.path.join(models_dir, selected_model)
        
        # Determine model architecture based on filename
        model_class, architecture_name = get_model_class_from_name(
            selected_model, 
            HybridRNNModel, 
            GRUGRUModel, 
            LSTMLSTMModel
        )
        print(f"\nDetected {architecture_name} architecture")
        
        # Get available tickers and select one
        tickers, counts = get_available_tickers(config['data_dir'])
        ticker = select_ticker(tickers, counts)
        
        # Create dataset for the selected ticker
        dataset = StockOptionDataset(
            data_dir=config['data_dir'],
            ticker=ticker,
            target_cols=config['target_cols']
        )
        
        logging.info(f"Running predictions with model: {selected_model}")
        run_existing_model(
            model_path,
            model_class,
            dataset,
            target_cols=config['target_cols']
        )
        logging.info("Predictions completed successfully")
    except Exception as e:
        logging.error(f"Error during model prediction: {str(e)}")
        print(f"\nError: {str(e)}")

def handle_run_model_enhanced(config, models_dir, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, 
                         get_available_tickers, select_ticker, StockOptionDataset, 
                         run_existing_model_with_visualization=None):
    """
    Enhanced handler for the model prediction workflow with visualization capability.
    
    Args:
        config: Application configuration
        models_dir: Directory containing saved models
        HybridRNNModel, GRUGRUModel, LSTMLSTMModel: Model classes
        get_available_tickers, select_ticker: Data utility functions
        StockOptionDataset: Dataset class
        run_existing_model_with_visualization: Function for running with visualization
    """
    try:
        import logging
        from pathlib import Path
        
        # Create visualization directory
        viz_dir = "prediction_visualizations"
        Path(viz_dir).mkdir(parents=True, exist_ok=True)
        
        # List available models
        model_files = list_available_models(models_dir)
        if not model_files:
            return

        # Select a model
        selected_model = select_model(model_files)
        if not selected_model:
            return

        model_path = os.path.join(models_dir, selected_model)
        
        # Determine model architecture based on filename
        model_class, architecture_name = get_model_class_from_name(
            selected_model, 
            HybridRNNModel, 
            GRUGRUModel, 
            LSTMLSTMModel
        )
        print(f"\nDetected {architecture_name} architecture")
        
        # Get available tickers and select one
        tickers, counts = get_available_tickers(config['data_dir'])
        ticker = select_ticker(tickers, counts)
        
        # Create dataset for the selected ticker
        dataset = StockOptionDataset(
            data_dir=config['data_dir'],
            ticker=ticker,
            target_cols=config['target_cols']
        )
        
        # Ask if user wants to visualize predictions
        visualize = input("\nVisualize predictions with original values? (y/n): ").lower().startswith('y')
        
        if visualize:
            # Ask for number of samples to visualize
            try:
                n_samples = int(input("\nNumber of samples to visualize (default: 100): ") or 100)
            except ValueError:
                n_samples = 100
                print("Invalid input. Using default value of 100 samples.")
                
            # Run the model with visualization
            logging.info(f"Running predictions with model: {selected_model} (with visualization)")
            
            if run_existing_model_with_visualization:
                errors, viz_files = run_existing_model_with_visualization(
                    model_path=model_path,
                    model_class=model_class,
                    dataset=dataset,
                    target_cols=config['target_cols'],
                    visualize=True,
                    n_samples=n_samples,
                    data_dir=config['data_dir'],
                    output_dir=viz_dir
                )
                
                print("\nPrediction visualization complete!")
                if viz_files and 'summary' in viz_files:
                    print(f"\nSummary visualization saved to: {viz_files['summary']}")
            else:
                # Fallback to standard run if visualization function not available
                print("\nVisualization function not available. Running standard prediction.")
                run_existing_model(
                    model_path,
                    model_class,
                    dataset,
                    target_cols=config['target_cols']
                )
        else:
            # Run standard prediction without visualization
            logging.info(f"Running predictions with model: {selected_model}")
            run_existing_model(
                model_path,
                model_class,
                dataset,
                target_cols=config['target_cols']
            )
            
        logging.info("Predictions completed successfully")
        
    except Exception as e:
        logging.error(f"Error during model prediction: {str(e)}")
        print(f"\nError: {str(e)}")

def handle_analyze_architecture(config, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, display_model_analysis):
    """Handle the model architecture analysis workflow."""
    try:
        # Allow user to select architecture to analyze
        print("\nSelect model architecture to analyze:")
        print("1. LSTM-GRU Hybrid (default)")
        print("2. GRU-GRU")
        print("3. LSTM-LSTM")
        
        arch_choice = input("\nEnter choice (1-3): ").strip()
        
        if arch_choice == "2":
            model = GRUGRUModel(
                input_size=23,
                hidden_size_gru1=config['hidden_size_gru'],
                hidden_size_gru2=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print("\nAnalyzing GRU-GRU architecture...")
        elif arch_choice == "3":
            model = LSTMLSTMModel(
                input_size=23,
                hidden_size_lstm1=config['hidden_size_lstm'],
                hidden_size_lstm2=config['hidden_size_lstm'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print("\nAnalyzing LSTM-LSTM architecture...")
        else:
            model = HybridRNNModel(
                input_size=23,
                hidden_size_lstm=config['hidden_size_lstm'],
                hidden_size_gru=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print("\nAnalyzing LSTM-GRU hybrid architecture...")
                
        analysis = analyze_model_architecture(model)
        display_model_analysis(analysis)
        logging.info("Architecture analysis completed")
    except Exception as e:
        logging.error(f"Error during architecture analysis: {str(e)}")
        print(f"\nError: {str(e)}")

def handle_benchmark_architectures(config, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, 
                                  benchmark_architectures, generate_architecture_comparison, 
                                  visualize_architectures, get_available_tickers, 
                                  select_ticker, StockOptionDataset):
    """Handle architecture benchmarking workflow with detailed progress output."""
    try:
        logging.info("Starting architecture benchmarking...")
        print("\n" + "="*70)
        print(f"{'BENCHMARKING MULTIPLE ARCHITECTURES':^70}")
        print("="*70)
        
        # Get ticker
        print("\n" + "-"*70)
        print("STEP 1: SELECT TICKER")
        print("-"*70)
        tickers, counts = get_available_tickers(config['data_dir'])
        ticker = select_ticker(tickers, counts)
        
        # Initialize dataset
        print("\n" + "-"*70)
        print("STEP 2: LOADING DATASET")
        print("-"*70)
        print(f"Loading dataset for {ticker}...")
        
        start_time = time.time()
        dataset = StockOptionDataset(
            data_dir=config['data_dir'], 
            ticker=ticker, 
            seq_len=config['seq_len'], 
            target_cols=config['target_cols']
        )
        
        if len(dataset) < 1:
            raise ValueError("Insufficient data for sequence creation!")
            
        print(f"✓ Dataset loaded successfully: {len(dataset):,} data points")
        print(f"✓ Input features: {dataset.n_features}")
        print(f"✓ Target columns: {', '.join(config['target_cols'])}")
        print(f"✓ Time taken: {time.time() - start_time:.2f} seconds")
            
        # Split dataset
        print("\n" + "-"*70)
        print("STEP 3: PREPARING DATA LOADERS")
        print("-"*70)
        
        print("Splitting dataset into train, validation, and test sets...")
        total_len = len(dataset)
        train_len = int(0.80 * total_len)
        val_len = int(0.10 * total_len)
        test_len = total_len - train_len - val_len
        
        indices = list(range(total_len))
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len+val_len]
        test_indices = indices[train_len+val_len:]
        
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices)
        
        print(f"✓ Training set: {len(train_ds):,} samples ({train_len/total_len*100:.1f}%)")
        print(f"✓ Validation set: {len(val_ds):,} samples ({val_len/total_len*100:.1f}%)")
        print(f"✓ Test set: {len(test_ds):,} samples ({test_len/total_len*100:.1f}%)")
        
        # Get benchmark parameters
        print("\n" + "-"*70)
        print("STEP 4: CONFIGURE BENCHMARK PARAMETERS")
        print("-"*70)
        
        batch_size = int(input("\nEnter batch size (default: 32): ") or "32")
        epochs = int(input("Enter number of epochs (default: 20): ") or "20")
        
        print("\nCreating data loaders...")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        print(f"✓ Data loaders created with batch size: {batch_size}")
        print(f"✓ Each epoch will process {len(train_loader)} training batches")
        
        # Create the models to benchmark
        print("\n" + "-"*70)
        print("STEP 5: INITIALIZING MODEL ARCHITECTURES")
        print("-"*70)
        
        hidden_size = config['hidden_size_lstm']  # Use same hidden size for fair comparison
        num_layers = config['num_layers']
        input_size = dataset.n_features
        output_size = len(config['target_cols'])
        
        print("\nInitializing LSTM-GRU Hybrid architecture...")
        lstm_gru_model = HybridRNNModel(
            input_size=input_size,
            hidden_size_lstm=hidden_size,
            hidden_size_gru=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        )
        lstm_gru_params = sum(p.numel() for p in lstm_gru_model.parameters())
        print(f"✓ LSTM-GRU model initialized with {lstm_gru_params:,} parameters")
        
        print("\nInitializing GRU-GRU architecture...")
        gru_gru_model = GRUGRUModel(
            input_size=input_size,
            hidden_size_gru1=hidden_size,
            hidden_size_gru2=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        )
        gru_gru_params = sum(p.numel() for p in gru_gru_model.parameters())
        print(f"✓ GRU-GRU model initialized with {gru_gru_params:,} parameters")
        
        print("\nInitializing LSTM-LSTM architecture...")
        lstm_lstm_model = LSTMLSTMModel(
            input_size=input_size,
            hidden_size_lstm1=hidden_size,
            hidden_size_lstm2=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        )
        lstm_lstm_params = sum(p.numel() for p in lstm_lstm_model.parameters())
        print(f"✓ LSTM-LSTM model initialized with {lstm_lstm_params:,} parameters")
        
        # Prepare model configurations
        models = [
            {'name': 'LSTM-GRU Hybrid', 'model': lstm_gru_model},
            {'name': 'GRU-GRU', 'model': gru_gru_model},
            {'name': 'LSTM-LSTM', 'model': lstm_lstm_model}
        ]
        
        print("\nAll architectures will be trained with:")
        print(f"- Hidden size: {hidden_size}")
        print(f"- Number of layers: {num_layers}")
        print(f"- Input features: {input_size}")
        print(f"- Output size: {output_size}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        
        # Run benchmarks
        print("\n" + "-"*70)
        print("STEP 6: RUNNING ARCHITECTURE BENCHMARKS")
        print("-"*70)
        
        print("\nStarting benchmark process for all architectures...")
        print("⚠ This may take considerable time. A detailed log will be created for each architecture.")
        print("\nTraining and evaluation sequence:")
        for i, model_config in enumerate(models, 1):
            print(f"  {i}. {model_config['name']}")
        
        benchmark_start_time = time.time()
        
        # First announce that we're starting benchmarks
        for i, model_config in enumerate(models, 1):
            arch_name = model_config['name']
            print(f"\n[{i}/{len(models)}] Benchmarking {arch_name}...")
            print("  This architecture will now be trained, validated, and tested.")
            print("  Progress will be shown for each epoch.")
            print("  A detailed performance log will be saved at the end.")
            print("  " + "-"*66)
            
        log_paths = benchmark_architectures(
            models=models,
            data_loaders=data_loaders,
            epochs=epochs,
            ticker=ticker,
            target_cols=config['target_cols'],
            save_dir=config['performance_logs_dir']
        )
        
        benchmark_time = time.time() - benchmark_start_time
        hours, remainder = divmod(benchmark_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "-"*70)
        print("STEP 7: GENERATING COMPARISON REPORT")
        print("-"*70)
        
        print(f"\nBenchmarking completed in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print("\nGenerating architecture comparison summary...")
        
        # Generate comparison summary
        summary_path = generate_architecture_comparison(log_paths)
        print(f"\n✓ Architecture comparison saved to: {summary_path}")
        
        print("\n" + "-"*70)
        print("STEP 8: CREATING VISUALIZATION CHARTS")
        print("-"*70)
        
        print("\nCreating performance visualization charts...")
        # Create visualizations
        viz_paths = visualize_architectures(log_paths, output_dir=config['performance_logs_dir'])
        
        if viz_paths:
            print(f"\n✓ Visualization charts created:")
            for i, path in enumerate(viz_paths, 1):
                print(f"  {i}. {path}")
            
        print("\n" + "="*70)
        print(f"{'BENCHMARKING COMPLETE':^70}")
        print("="*70)
        print("\nYou can now analyze the results to determine which architecture")
        print("performs best for your specific option pricing task.")
                
        logging.info("Architecture benchmarking completed successfully")
        
    except Exception as e:
        logging.error(f"Error during architecture benchmarking: {str(e)}")
        print(f"\n❌ Error: {str(e)}")

def debug_scaling_parameters(ticker: str, data_dir: str) -> None:
    """
    Debug function to examine scaling parameters file for a specific ticker.
    
    Args:
        ticker: Ticker symbol
        data_dir: Data directory path
    """
    try:
        # Try different possible paths
        possible_paths = [
            os.path.join(data_dir, 'by_ticker', 'scaling_params', f"{ticker}_scaling_params.json"),
            os.path.join(data_dir, 'scaling_params', f"{ticker}_scaling_params.json"),
            os.path.join(data_dir, f"{ticker}_scaling_params.json")
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"\nFound scaling parameters at: {path}")
                with open(path, 'r') as f:
                    params = json.load(f)
                
                # Print target column parameters if they exist
                target_cols = ['bid', 'ask']  # Adjust as needed
                for col in target_cols:
                    if col in params:
                        print(f"Parameters for {col}: {params[col]}")
                    else:
                        print(f"Column {col} not found in parameters")
                
                # Print a sample of 5 columns
                print("\nSample of scaling parameters (5 columns):")
                for i, (col, values) in enumerate(params.items()):
                    if i >= 5:
                        break
                    print(f"  {col}: {values}")
                
                found = True
                break
        
        if not found:
            print("\nNo scaling parameters file found at any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            
            # Check directory structure
            parent_dir = os.path.dirname(possible_paths[0])
            if os.path.exists(parent_dir):
                print(f"\nContents of directory {parent_dir}:")
                for item in os.listdir(parent_dir):
                    print(f"  - {item}")
            else:
                print(f"\nDirectory does not exist: {parent_dir}")
                
    except Exception as e:
        print(f"Error examining scaling parameters: {str(e)}")