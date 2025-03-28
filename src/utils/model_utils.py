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

def analyze_model_architecture(model, input_size=23, seq_len=15, batch_size=32):
    """Analyze the architecture of the model, including parameter count and tensor shapes."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    layer_shapes = {}
    
    def hook_fn(module, input, output, name):
        def get_tensor_shape(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            elif isinstance(x, tuple):
                return tuple(get_tensor_shape(t) for t in x if isinstance(t, torch.Tensor))
            return None

        layer_shapes[name] = {
            'input_shape': [tuple(i.shape) for i in input],
            'output_shape': get_tensor_shape(output)
        }
    
    hooks = []
    for name, layer in model.named_children():
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, name=name: hook_fn(m, i, o, name)
        ))
    
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layer_shapes': layer_shapes
    }

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
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
    model_name = model_name.lower()
    if "gru-gru" in model_name:
        return GRUGRUModel, "GRU-GRU"
    elif "lstm-lstm" in model_name:
        return LSTMLSTMModel, "LSTM-LSTM"
    else:
        return HybridRNNModel, "LSTM-GRU"
        
def load_model(model_path, model_class, input_size, 
               hidden_size_lstm=128, hidden_size_gru=128, 
               hidden_size_lstm1=128, hidden_size_lstm2=128,
               hidden_size_gru1=128, hidden_size_gru2=128,
               num_layers=2, output_size=1):
    """
    Load a saved model with proper architecture class.
    Supports all three architecture types.
    """
    # Import needed here to avoid circular import
    from nn import HybridRNNModel, GRUGRUModel, LSTMLSTMModel
    
    if model_class == HybridRNNModel:
        model = model_class(
            input_size=input_size,
            hidden_size_lstm=hidden_size_lstm,
            hidden_size_gru=hidden_size_gru,
            num_layers=num_layers,
            output_size=output_size
        )
    elif model_class == GRUGRUModel:
        model = model_class(
            input_size=input_size,
            hidden_size_gru1=hidden_size_gru1,
            hidden_size_gru2=hidden_size_gru2,
            num_layers=num_layers,
            output_size=output_size
        )
    elif model_class == LSTMLSTMModel:
        model = model_class(
            input_size=input_size,
            hidden_size_lstm1=hidden_size_lstm1,
            hidden_size_lstm2=hidden_size_lstm2,
            num_layers=num_layers,
            output_size=output_size
        )
    
    model.load_state_dict(torch.load(model_path))
    return model

def run_existing_model(model_path, model_class, dataset, target_cols=["bid", "ask"]):
    """Load and run predictions with an existing model."""
    model = load_model(
        model_path, 
        model_class,
        input_size=dataset.n_features, 
        output_size=len(dataset.target_cols)
    )
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

def load_scaling_params(ticker: str, data_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load the scaling parameters for a specific ticker.
    
    Args:
        ticker: The ticker symbol
        data_dir: Directory containing data files
        
    Returns:
        Dictionary of scaling parameters
    """
    params_path = os.path.join(data_dir, 'by_ticker', 'scaling_params', f"{ticker}_scaling_params.json")
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Scaling parameters not found for {ticker} at {params_path}")
    
    with open(params_path, 'r') as f:
        scaling_params = json.load(f)
    
    return scaling_params

def recover_original_values(normalized_values: np.ndarray, 
                          column_names: List[str],
                          scaling_params: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Recover original values from normalized values using scaling parameters.
    
    Args:
        normalized_values: Array of normalized values
        column_names: List of column names corresponding to the values
        scaling_params: Dictionary of scaling parameters
        
    Returns:
        Array with recovered original values
    """
    original_values = normalized_values.copy()
    
    for i, col in enumerate(column_names):
        if col in scaling_params:
            # Get parameters for this column
            mean = scaling_params[col]['mean'] if 'mean' in scaling_params[col] else scaling_params[col].get('mean', 0)
            scale = scaling_params[col]['scale'] if 'scale' in scaling_params[col] else scaling_params[col].get('std', 1)
            
            # Apply inverse transform: X_orig = X_scaled * scale + mean
            original_values[:, i] = normalized_values[:, i] * scale + mean
    
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
    dataset: Any, 
    target_cols: List[str] = ["bid", "ask"],
    visualize: bool = True,
    n_samples: int = 100,
    data_dir: Optional[str] = None,
    output_dir: str = "prediction_visualizations"
) -> Tuple[Dict[str, float], Optional[Dict[str, str]]]:
    """
    Load and run predictions with an existing model, with optional visualization.
    
    Args:
        model_path: Path to the saved model
        model_class: Model class to use
        dataset: Dataset containing the features and targets
        target_cols: Target columns being predicted
        visualize: Whether to create visualization plots
        n_samples: Number of samples to visualize
        data_dir: Path to data directory (needed for scaling parameters)
        output_dir: Directory to save visualizations
        
    Returns:
        Tuple of (error metrics, visualization file paths)
    """
    model = load_model(
        model_path, 
        model_class,
        input_size=dataset.n_features, 
        output_size=len(dataset.target_cols)
    )
    model.eval()
    
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"\nMaking predictions for {dataset.ticker}...")
    with torch.no_grad():
        for x_seq, y_val in data_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            y_pred = model(x_seq)
            all_y_true.extend(y_val.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())
    
    # Calculate error metrics
    errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
    
    # Print error metrics
    print("\nPrediction Metrics:")
    print("-" * 50)
    print(f"MSE: {errors['mse']:.6f}")
    print(f"RMSE: {errors['rmse']:.6f}")
    print(f"MAE: {errors['mae']:.6f}")
    print(f"MAPE: {errors['mape']:.2f}%")
    
    # Create visualizations if requested
    viz_files = None
    if visualize:
        # Try to load scaling parameters if data_dir is provided
        scaling_params = None
        if data_dir:
            try:
                scaling_params = load_scaling_params(dataset.ticker, data_dir)
                print(f"\nLoaded scaling parameters for {dataset.ticker} for visualization with original values")
            except FileNotFoundError:
                print(f"\nScaling parameters not found for {dataset.ticker}. Using normalized values for visualization.")
        
        # Create visualizations
        viz_files = visualize_predictions(
            y_true=np.array(all_y_true),
            y_pred=np.array(all_y_pred),
            target_cols=target_cols,
            ticker=dataset.ticker,
            scaling_params=scaling_params,
            output_dir=output_dir,
            show_plot=True,
            n_samples=n_samples
        )
        
        # Print visualization file paths
        print("\nVisualizations saved to:")
        for target, file_path in viz_files.items():
            print(f"  {target}: {file_path}")
    
    return errors, viz_files
# Handler Functions moved from nn.py
def handle_train_model(config, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, save_and_display_results, extended_train_model_with_tracking, get_available_tickers, select_ticker, StockOptionDataset):
    """Handle the model training workflow."""
    try:
        logging.info("Starting model training...")
        # Ask user if they want to use performance tracking
        use_tracking = input("\nUse performance tracking? (y/n): ").lower().startswith('y')
        
        # Ask user to select model architecture
        print("\nSelect model architecture:")
        print("1. LSTM-GRU Hybrid (default)")
        print("2. GRU-GRU")
        print("3. LSTM-LSTM")
        
        arch_choice = input("\nEnter choice (1-3): ").strip()
        if arch_choice == "2":
            architecture_type = "GRU-GRU"
        elif arch_choice == "3":
            architecture_type = "LSTM-LSTM"
        else:
            architecture_type = "LSTM-GRU"
        
        # Get available tickers and select one
        tickers, counts = get_available_tickers(config['data_dir'])
        ticker = select_ticker(tickers, counts)
        
        # Initialize dataset
        dataset = StockOptionDataset(
            data_dir=config['data_dir'], 
            ticker=ticker, 
            seq_len=config['seq_len'], 
            target_cols=config['target_cols']
        )
        
        if len(dataset) < 1:
            raise ValueError("Insufficient data for sequence creation!")
        
        # Split dataset maintaining temporal order
        total_len = len(dataset)
        train_len = int(0.80 * total_len)
        val_len = int(0.10 * total_len)
        
        indices = list(range(total_len))
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len+val_len]
        test_indices = indices[train_len+val_len:]
        
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model based on selected architecture
        if architecture_type == "GRU-GRU":
            model = GRUGRUModel(
                input_size=dataset.n_features,
                hidden_size_gru1=config['hidden_size_gru'],
                hidden_size_gru2=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print(f"\nInitialized GRU-GRU architecture")
        elif architecture_type == "LSTM-LSTM":
            model = LSTMLSTMModel(
                input_size=dataset.n_features,
                hidden_size_lstm1=config['hidden_size_lstm'],
                hidden_size_lstm2=config['hidden_size_lstm'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print(f"\nInitialized LSTM-LSTM architecture")
        else:  # Default to LSTM-GRU hybrid
            model = HybridRNNModel(
                input_size=dataset.n_features,
                hidden_size_lstm=config['hidden_size_lstm'],
                hidden_size_gru=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print(f"\nInitialized LSTM-GRU hybrid architecture")
        
        model_analysis = analyze_model_architecture(
            model, 
            input_size=dataset.n_features,
            seq_len=config['seq_len']
        )
        
        if use_tracking:
            # Use performance tracking training
            model, history, log_path = extended_train_model_with_tracking(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=config['epochs'],
                lr=1e-3,
                device=device,
                ticker=ticker,
                architecture_name=architecture_type,
                target_cols=config['target_cols']
            )
            print(f"\nPerformance log saved to: {log_path}")
        else:
            # Standard training without performance tracking
            train_losses, val_losses = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config['epochs'],
                lr=1e-3,
                device=device
            )
            
            history = {'train_losses': train_losses, 'val_losses': val_losses}
            
            # Final evaluation
            model.eval()
            test_loss = 0.0
            all_y_true = []
            all_y_pred = []
            criterion = nn.MSELoss()
            
            with torch.no_grad():
                for x_seq, y_val in test_loader:
                    x_seq, y_val = x_seq.to(device), y_val.to(device)
                    y_pred = model(x_seq)
                    loss = criterion(y_pred, y_val)
                    test_loss += loss.item()
                    
                    all_y_true.extend(y_val.cpu().numpy())
                    all_y_pred.extend(y_pred.cpu().numpy())
            
            errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
            print("\nFinal Test Set Metrics:")
            print("-" * 50)
            print(f"MSE: {errors['mse']:.6f}")
            print(f"RMSE: {errors['rmse']:.6f}")
            print(f"MAE: {errors['mae']:.6f}")
            print(f"MAPE: {errors['mape']:.2f}%")
            
        save_and_display_results(model, history, model_analysis, ticker, config['target_cols'], models_dir=config['models_dir'])
        logging.info("Model training completed successfully")
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        print(f"\nError: {str(e)}")

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