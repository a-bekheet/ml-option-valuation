import torch
import numpy as np
import time
import os
import psutil
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from utils.visualization_utils import plot_predictions
from torch.utils.data import DataLoader
from utils.model_utils import EarlyStopping, calculate_errors

def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the directional accuracy (percentage of correctly predicted price movements).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy as a percentage
    """
    # Convert arrays to ensure proper shape
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate the direction of movement (up or down)
    if len(y_true) <= 1:
        return np.nan
    
    y_true_direction = np.diff(y_true, axis=0) > 0
    y_pred_direction = np.diff(y_pred, axis=0) > 0
    
    # Calculate percentage of correct directional predictions
    correct_predictions = np.sum(y_true_direction == y_pred_direction)
    total_predictions = y_true_direction.size
    
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def calculate_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the maximum prediction error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Maximum absolute error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.max(np.abs(y_true - y_pred)))

def track_performance(model, train_loader, val_loader, test_loader,
                     epochs: int, ticker: str, architecture_name: str,
                     target_cols: List[str],
                     # >> NEW: Add parameters for features and architecture <<
                     used_features: Optional[List[str]] = None,
                     model_analysis_dict: Optional[Dict[str, Any]] = None,
                     # >> END NEW <<
                     save_dir: str = "performance_logs",
                     verbose: bool = True,
                     lr: float = 0.001) -> Tuple[str, Dict]: # Return log_path and history
    """
    Track and log comprehensive model performance metrics, including features and architecture.

    Args:
        model: PyTorch model to evaluate
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        ticker: Stock ticker symbol
        architecture_name: Name of the model architecture
        target_cols: Target columns being predicted
        used_features (Optional[List[str]]): List of feature names used in this run.
        model_analysis_dict (Optional[Dict[str, Any]]): Dictionary containing model architecture analysis.
        save_dir (str): Directory to save performance logs
        verbose (bool): Whether to print detailed progress to terminal
        lr (float): Learning rate for the optimizer.

    Returns:
        Tuple[str, Dict]: Path to the created log file, history dictionary
    """
    # Create log directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{architecture_name}_{ticker}_{timestamp}.txt"
    log_path = os.path.join(save_dir, log_filename)

    # Get device
    device = next(model.parameters()).device
    history = {'train_losses': [], 'val_losses': [], 'test_metrics': {}, 'epoch_times': []} # Initialize history

    # Prepare log file
    with open(log_path, 'w') as f:
        # Write header information
        f.write(f"{'='*80}\n")
        f.write(f"MODEL PERFORMANCE REPORT\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Model Architecture: {architecture_name}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Target Columns: {', '.join(target_cols)}\n")
        start_run_time = datetime.datetime.now()
        f.write(f"Run Started: {start_run_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Max Epochs: {epochs}\n\n")


        # >> NEW: Log Features Used <<
        if used_features:
             f.write(f"{'-'*80}\n")
             f.write(f"FEATURES USED ({len(used_features)})\n")
             f.write(f"{'-'*80}\n")
             # Write features, possibly wrapping lines for readability
             feature_lines = []
             current_line = ""
             for feature in used_features:
                  if not current_line:
                       current_line = feature
                  elif len(current_line) + len(feature) + 2 < 100: # Max line length approx 100
                       current_line += f", {feature}"
                  else:
                       feature_lines.append(current_line)
                       current_line = feature
             if current_line: feature_lines.append(current_line) # Add last line
             f.write("\n".join(feature_lines) + "\n\n")
        else:
             f.write("Features used: Not Provided\n\n")
        # >> END NEW <<

        # >> NEW: Log Model Architecture <<
        if model_analysis_dict:
            f.write(f"{'-'*80}\n")
            f.write(f"MODEL ARCHITECTURE DETAILS\n")
            f.write(f"{'-'*80}\n")
            total_params = model_analysis_dict.get('total_parameters', 'N/A')
            trainable_params = model_analysis_dict.get('trainable_parameters', 'N/A')
            f.write(f"Total Parameters: {total_params:,}\n" if isinstance(total_params, int) else f"Total Parameters: {total_params}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n\n" if isinstance(trainable_params, int) else f"Trainable Parameters: {trainable_params}\n\n")
            f.write("Layer Shapes (Input -> Output):\n")
            layer_shapes = model_analysis_dict.get('layer_shapes', {})
            if layer_shapes:
                 for layer_name, shapes in layer_shapes.items():
                      in_shape = shapes.get('input_shape', 'N/A')
                      out_shape = shapes.get('output_shape', 'N/A')
                      f.write(f"  {layer_name}:\n")
                      # Format shapes for better readability if they are tuples/lists
                      f.write(f"    Input: {in_shape}\n")
                      f.write(f"    Output: {out_shape}\n")
            else:
                 f.write("  Layer shape details not available.\n")
            f.write("\n") # Add space after architecture section
        else:
             # Fallback: Log basic param count if full analysis missing
             try:
                  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                  total_params = sum(p.numel() for p in model.parameters())
                  f.write(f"{'-'*80}\n")
                  f.write(f"MODEL COMPLEXITY (Basic)\n")
                  f.write(f"{'-'*80}\n")
                  f.write(f"Trainable Parameters: {trainable_params:,}\n")
                  f.write(f"Total Parameters: {total_params:,}\n\n")
             except Exception:
                  f.write("Model Analysis: Not Provided\n\n")
        # >> END NEW <<


        # Training metrics section header
        f.write(f"{'-'*80}\n")
        f.write(f"TRAINING METRICS\n")
        f.write(f"{'-'*80}\n\n")

        # Prepare for training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Added weight decay
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=verbose # Made scheduler verbose
        )
        early_stopping = EarlyStopping(patience=5, min_delta=1e-5) # Use configured EarlyStopping

        # Training loop header
        f.write(f"{'Epoch':^6}|{'Train Loss':^12}|{'Val Loss':^12}|{'LR':^10}|{'Time (s)':^10}|{'Memory (MB)':^12}\n")
        f.write(f"{'-'*6}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*12}\n")

        if verbose:
            print(f"\n{'-' * 80}")
            print(f"{architecture_name} - TRAINING PHASE")
            print(f"{'-' * 80}")
            try: # Add try-except for dataloader length issues
                 print(f"Train samples: {len(train_loader.dataset):,}, Val samples: {len(val_loader.dataset):,}, Test samples: {len(test_loader.dataset):,}")
                 print(f"Batches per epoch: {len(train_loader):,}")
            except TypeError:
                 print("Could not determine dataset lengths (possibly using Subset without full dataset access).")
            print(f"Device: {device}")
            print(f"{'-' * 50}")

        total_train_time = 0.0
        actual_epochs_run = 0

        for epoch in range(epochs):
            actual_epochs_run += 1
            epoch_start_time = time.time()

            # --- Training phase ---
            model.train()
            total_train_loss = 0.0
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch", disable=not verbose, leave=False)
            for x_seq, y_val in train_iter:
                x_seq, y_val = x_seq.to(device), y_val.to(device)
                optimizer.zero_grad()
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_val)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
                if verbose: train_iter.set_postfix({'loss': f'{loss.item():.6f}'})

            avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            history['train_losses'].append(avg_train_loss)

            # --- Validation phase ---
            model.eval()
            total_val_loss = 0.0
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", unit="batch", disable=not verbose, leave=False)
            with torch.no_grad():
                for x_seq, y_val in val_iter:
                    x_seq, y_val = x_seq.to(device), y_val.to(device)
                    y_pred = model(x_seq)
                    loss = criterion(y_pred, y_val)
                    total_val_loss += loss.item()
                    if verbose: val_iter.set_postfix({'val_loss': f'{loss.item():.6f}'})

            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            history['val_losses'].append(avg_val_loss)

            # Scheduler and Early Stopping
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss)

            # Record epoch time and memory
            epoch_time = time.time() - epoch_start_time
            history['epoch_times'].append(epoch_time)
            total_train_time += epoch_time
            try: # Handle potential errors getting memory info
                 if str(device).startswith('cuda'):
                      memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024) # MB
                 elif str(device).startswith('mps'):
                      # MPS memory reporting is complex; use process RSS as approximation
                      memory_allocated = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) # MB
                 else: # CPU
                      memory_allocated = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) # MB
            except Exception as mem_err:
                 logging.warning(f"Could not get memory usage: {mem_err}")
                 memory_allocated = np.nan

            # Log epoch results
            current_lr = optimizer.param_groups[0]['lr']
            f.write(f"{epoch+1:^6}|{avg_train_loss:^12.6f}|{avg_val_loss:^12.6f}|{current_lr:^10.2e}|{epoch_time:^10.2f}|{memory_allocated:^12.2f}\n")

            if verbose:
                print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
                print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.2f}s | Memory: {memory_allocated:.2f} MB")

            # Check for early stopping
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                f.write("\nEarly stopping triggered.\n")
                break

        # --- Training Summary ---
        avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times']) if history['epoch_times'] else 0
        f.write(f"\nTraining Summary:\n")
        f.write(f"Training stopped after epoch {actual_epochs_run}\n")
        f.write(f"Total Training Time: {total_train_time:.2f} seconds\n")
        f.write(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds\n")
        f.write(f"Best Validation Loss: {early_stopping.best_loss:.6f}\n" if early_stopping.best_loss is not None else "Best Validation Loss: N/A\n")


        # --- Test Set Evaluation ---
        f.write(f"\n{'-'*80}\n")
        f.write(f"TEST SET EVALUATION\n")
        f.write(f"{'-'*80}\n\n")

        if verbose:
            print(f"\n{'-' * 80}")
            print(f"{architecture_name} - TESTING PHASE")
            print(f"{'-' * 80}")

        inference_start_time = time.time()
        model.eval()
        test_loss = 0.0
        all_y_true_test = []
        all_y_pred_test = []

        with torch.no_grad():
            test_iter = tqdm(test_loader, desc="Testing", unit="batch", disable=not verbose)
            for x_seq, y_true in test_iter:
                x_seq, y_true = x_seq.to(device), y_true.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_true)
                test_loss += loss.item()
                all_y_true_test.extend(y_true.cpu().numpy())
                all_y_pred_test.extend(y_pred.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        inference_time = time.time() - inference_start_time
        total_test_samples = len(all_y_true_test)
        inference_time_per_sample = (inference_time / total_test_samples * 1000) if total_test_samples > 0 else 0 # in ms

        # Calculate final metrics on test set
        y_true_np = np.array(all_y_true_test)
        y_pred_np = np.array(all_y_pred_test)
        test_errors = calculate_errors(torch.tensor(y_true_np), torch.tensor(y_pred_np)) # Reuse existing function
        history['test_metrics'] = test_errors # Store metrics in history

        # Calculate additional metrics
        if total_test_samples > 1:
             directional_accuracy = calculate_directional_accuracy(y_true_np, y_pred_np)
             history['test_metrics']['directional_accuracy'] = directional_accuracy
             max_error = calculate_max_error(y_true_np, y_pred_np)
             history['test_metrics']['max_error'] = max_error
        else:
             directional_accuracy = np.nan
             max_error = np.nan


        # Log test metrics
        f.write(f"Test Loss (MSE): {avg_test_loss:.6f}\n")
        f.write(f"Test RMSE: {test_errors.get('rmse', np.nan):.6f}\n")
        f.write(f"Test MAE: {test_errors.get('mae', np.nan):.6f}\n")
        f.write(f"Test MAPE: {test_errors.get('mape', np.nan):.2f}%\n")
        f.write(f"Test Directional Accuracy: {directional_accuracy:.2f}%\n")
        f.write(f"Test Maximum Error: {max_error:.6f}\n\n")
        f.write(f"Total Inference Time (Test Set): {inference_time:.4f} seconds\n")
        f.write(f"Inference Time per Sample: {inference_time_per_sample:.4f} ms\n\n")

        # Add test predictions to history for potential plotting
        history['y_true'] = all_y_true_test
        history['y_pred'] = all_y_pred_test

        # --- Generate Prediction Plots (Optional but Recommended) ---
        if total_test_samples > 0:
             try:
                 plots_dir = os.path.join(save_dir, 'plots')
                 Path(plots_dir).mkdir(parents=True, exist_ok=True)
                 plot_timestamp = timestamp # Use same timestamp as log file
                 if verbose: print(f"\nGenerating prediction plots...")

                 # Call visualization function (ensure it's imported)
                 plot_predictions(y_true_np, y_pred_np, target_cols, ticker, plots_dir, plot_timestamp)
                 pred_plot_path = os.path.join(plots_dir, f"{ticker}_predictions_{plot_timestamp}.png")
                 scatter_plot_path = os.path.join(plots_dir, f"{ticker}_pred_scatter_{plot_timestamp}.png")

                 f.write(f"Prediction Plots:\n")
                 f.write(f"  Time Series Plot: {pred_plot_path}\n")
                 f.write(f"  Scatter Plot: {scatter_plot_path}\n\n")
                 if verbose: print(f"Prediction plots saved to {plots_dir}")
             except Exception as plot_err:
                 logging.error(f"Failed to generate prediction plots: {plot_err}")
                 f.write("Failed to generate prediction plots.\n\n")
        else:
             f.write("No test samples available to generate prediction plots.\n\n")


        # --- Closing ---
        end_run_time = datetime.datetime.now()
        duration = end_run_time - start_run_time
        f.write(f"Run Completed: {end_run_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {str(duration).split('.')[0]}\n") # Format duration cleanly

        # Print final summary if verbose
        if verbose:
            print(f"\n{'-' * 80}")
            print("TEST SET FINAL METRICS")
            print(f"{'-' * 30}")
            print(f"Test Loss (MSE): {avg_test_loss:.6f}")
            print(f"Test RMSE: {test_errors.get('rmse', np.nan):.6f}")
            print(f"Test MAE: {test_errors.get('mae', np.nan):.6f}")
            print(f"Test MAPE: {test_errors.get('mape', np.nan):.2f}%")
            print(f"Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"Maximum Error: {max_error:.6f}")
            print(f"Inference Time per Sample: {inference_time_per_sample:.4f} ms")
            print(f"\nTotal Run Duration: {str(duration).split('.')[0]}")
            print(f"Log saved to: {log_path}")
            print(f"{'-' * 80}\n")

    logging.info(f"Performance metrics saved to {log_path}")
    return log_path, history # Return log path and history dict

def benchmark_architectures(models: List[Dict[str, Any]], 
                          data_loaders: Dict[str, Any],
                          epochs: int, 
                          ticker: str,
                          target_cols: List[str], 
                          save_dir: str = "performance_logs") -> List[str]:
    """
    Compare multiple model architectures and log their performance.
    
    Args:
        models: List of model configuration dictionaries with 'name' and 'model' keys
        data_loaders: Dictionary with 'train', 'val', and 'test' data loaders
        epochs: Number of training epochs
        ticker: Stock ticker symbol
        target_cols: Target columns being predicted
        save_dir: Directory to save performance logs
        
    Returns:
        List of paths to the created log files
    """
    log_paths = []
    total_models = len(models)
    
    # Print benchmark header
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK COMPARISON: {total_models} ARCHITECTURES ON {ticker}")
    print(f"{'=' * 80}")
    print(f"Target columns: {', '.join(target_cols)}")
    print(f"Training for {epochs} epochs per architecture")
    print(f"Training samples: {len(data_loaders['train'].dataset):,}")
    print(f"Validation samples: {len(data_loaders['val'].dataset):,}")
    print(f"Test samples: {len(data_loaders['test'].dataset):,}")
    
    # Record start time for entire benchmark
    benchmark_start = time.time()
    
    # Run each model
    for i, model_config in enumerate(models, 1):
        model_name = model_config['name']
        model = model_config['model']
        
        # Determine device to use
        device = 'cuda' if torch.cuda.is_available() else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
        model = model.to(device)
        
        # Clear separator between models
        print(f"\n{'=' * 80}")
        print(f"[{i}/{total_models}] BENCHMARKING: {model_name}")
        print(f"{'=' * 80}")
        
        # Display parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {param_count:,}")
        print(f"Using device: {device}")
        
        # Record start time for this model
        model_start = time.time()
        
        # Run performance tracking with verbose output
        log_path = track_performance(
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            test_loader=data_loaders['test'],
            epochs=epochs,
            ticker=ticker,
            architecture_name=model_name,
            target_cols=target_cols,
            save_dir=save_dir,
            verbose=True  # Enable verbose output
        )
        
        log_paths.append(log_path)
        
        # Calculate and display time taken for this model
        model_time = time.time() - model_start
        hours, remainder = divmod(model_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nCompleted {model_name} in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print(f"Performance log saved to: {log_path}")
        
        # Show progress through benchmark
        if i < total_models:
            remaining = total_models - i
            print(f"\n{remaining} architecture(s) remaining...\n")
    
    # Calculate and display total benchmark time
    benchmark_time = time.time() - benchmark_start
    hours, remainder = divmod(benchmark_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Generated {len(log_paths)} performance logs")
    
    return log_paths

def generate_architecture_comparison(log_paths: List[str], output_path: Optional[str] = None) -> str:
    """
    Generate a summary of performance metrics from multiple architecture log files.
    
    Args:
        log_paths: List of paths to performance log files
        output_path: Path to save the summary (if None, a default path is used)
        
    Returns:
        Path to the created summary file
    """
    if not log_paths:
        raise ValueError("No log paths provided for summary generation")
    
    # Create default output path if not provided
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"performance_logs/architecture_comparison_{timestamp}.txt"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract data from log files
    architectures = []
    metrics = {
        'params': [],
        'train_time': [],
        'mse': [],
        'rmse': [],
        'directional_accuracy': [],
        'max_error': [],
        'inference_time': []
    }
    
    for log_path in log_paths:
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                
                # Extract architecture name
                arch_match = content.split("Model Architecture: ")[1].split("\n")[0] if "Model Architecture: " in content else "Unknown"
                architectures.append(arch_match)
                
                # Extract trainable parameters
                params_match = content.split("Trainable Parameters: ")[1].split("\n")[0] if "Trainable Parameters: " in content else "N/A"
                metrics['params'].append(params_match)
                
                # Extract average training time
                train_time_match = content.split("Average Time per Epoch: ")[1].split(" seconds")[0] if "Average Time per Epoch: " in content else "N/A"
                metrics['train_time'].append(train_time_match)
                
                # Extract MSE
                mse_match = content.split("MSE: ")[1].split("\n")[0] if "MSE: " in content else "N/A"
                metrics['mse'].append(mse_match)
                
                # Extract RMSE
                rmse_match = content.split("RMSE: ")[1].split("\n")[0] if "RMSE: " in content else "N/A"
                metrics['rmse'].append(rmse_match)
                
                # Extract directional accuracy
                dir_acc_match = content.split("Directional Accuracy: ")[1].split("%")[0] if "Directional Accuracy: " in content else "N/A"
                metrics['directional_accuracy'].append(dir_acc_match)
                
                # Extract maximum error
                max_error_match = content.split("Maximum Error: ")[1].split("\n")[0] if "Maximum Error: " in content else "N/A"
                metrics['max_error'].append(max_error_match)
                
                # Extract inference time
                inf_time_match = content.split("Inference Time per Sample: ")[1].split(" ms")[0] if "Inference Time per Sample: " in content else "N/A"
                metrics['inference_time'].append(inf_time_match)
                
        except Exception as e:
            logging.error(f"Error parsing log file {log_path}: {str(e)}")
            continue
    
    print(f"\nGenerating comparison summary for {len(architectures)} architectures...")
    
    # Write summary to file
    with open(output_path, 'w') as f:
        f.write(f"{'='*100}\n")
        f.write(f"{'ARCHITECTURE PERFORMANCE COMPARISON':^100}\n")
        f.write(f"{'='*100}\n\n")
        
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write comparison table
        f.write(f"{'Architecture':<20}|{'Parameters':<15}|{'Avg Epoch(s)':<12}|{'MSE':<10}|{'RMSE':<10}|{'Dir Acc(%)':<10}|{'Max Error':<10}|{'Infer(ms)':<10}\n")
        f.write(f"{'-'*20}|{'-'*15}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}\n")
        
        for i, arch in enumerate(architectures):
            f.write(f"{arch:<20}|{metrics['params'][i]:<15}|{metrics['train_time'][i]:<12}|{metrics['mse'][i]:<10}|{metrics['rmse'][i]:<10}|{metrics['directional_accuracy'][i]:<10}|{metrics['max_error'][i]:<10}|{metrics['inference_time'][i]:<10}\n")
        
        f.write(f"\nDetailed logs for each architecture are available in the performance_logs directory.\n")
    
    logging.info(f"Architecture comparison saved to {output_path}")
    return output_path

def visualize_architectures(log_paths: List[str], output_dir: str = "performance_logs") -> List[str]:
    """
    Create visualizations comparing different model architectures.
    
    Args:
        log_paths: List of paths to performance log files
        output_dir: Directory to save visualizations
        
    Returns:
        List of paths to the created visualization files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract data for visualizations
    architectures = []
    mse_values = []
    rmse_values = []
    dir_acc_values = []
    max_error_values = []
    train_times = []
    inference_times = []
    param_counts = []
    
    for log_path in log_paths:
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                
                arch_name = content.split("Model Architecture: ")[1].split("\n")[0] if "Model Architecture: " in content else "Unknown"
                architectures.append(arch_name)
                
                # Extract metrics
                mse = float(content.split("MSE: ")[1].split("\n")[0]) if "MSE: " in content else 0
                mse_values.append(mse)
                
                rmse = float(content.split("RMSE: ")[1].split("\n")[0]) if "RMSE: " in content else 0
                rmse_values.append(rmse)
                
                dir_acc = float(content.split("Directional Accuracy: ")[1].split("%")[0]) if "Directional Accuracy: " in content else 0
                dir_acc_values.append(dir_acc)
                
                max_error = float(content.split("Maximum Error: ")[1].split("\n")[0]) if "Maximum Error: " in content else 0
                max_error_values.append(max_error)
                
                train_time = float(content.split("Average Time per Epoch: ")[1].split(" seconds")[0]) if "Average Time per Epoch: " in content else 0
                train_times.append(train_time)
                
                inf_time = float(content.split("Inference Time per Sample: ")[1].split(" ms")[0]) if "Inference Time per Sample: " in content else 0
                inference_times.append(inf_time)
                
                params = int(content.split("Trainable Parameters: ")[1].split("\n")[0].replace(",", "")) if "Trainable Parameters: " in content else 0
                param_counts.append(params)
                
        except Exception as e:
            logging.error(f"Error extracting data from log file {log_path}: {str(e)}")
            continue
    
    if not architectures:
        logging.warning("No valid architecture data found for visualization")
        return []
    
    saved_paths = []
    
    print(f"\nCreating visualization charts for {len(architectures)} architectures...")
    
    # Set style
    plt.style.use('ggplot')
    
    # Figure 1: Prediction Error Metrics
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(architectures, mse_values)
    plt.title('MSE by Architecture')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(architectures, rmse_values)
    plt.title('RMSE by Architecture')
    plt.ylabel('Root Mean Squared Error')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 3)
    bars = plt.bar(architectures, dir_acc_values)
    plt.title('Directional Accuracy by Architecture')
    plt.ylabel('Directional Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 4)
    bars = plt.bar(architectures, max_error_values)
    plt.title('Maximum Error by Architecture')
    plt.ylabel('Maximum Error')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    error_metrics_path = os.path.join(output_dir, f"architecture_error_metrics_{timestamp}.png")
    plt.savefig(error_metrics_path, dpi=300)
    plt.close()
    saved_paths.append(error_metrics_path)
    
    # Figure 2: Efficiency Metrics
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(architectures, train_times)
    plt.title('Training Time by Architecture')
    plt.ylabel('Avg Time per Epoch (seconds)')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(architectures, inference_times)
    plt.title('Inference Time by Architecture')
    plt.ylabel('Inference Time per Sample (ms)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    efficiency_path = os.path.join(output_dir, f"architecture_efficiency_{timestamp}.png")
    plt.savefig(efficiency_path, dpi=300)
    plt.close()
    saved_paths.append(efficiency_path)
    
    # Figure 3: Complexity vs Performance
    plt.figure(figsize=(10, 8))
    
    # Normalize parameter counts for bubble size
    max_params = max(param_counts) if param_counts else 1
    normalized_params = [100 + (p / max_params) * 1000 for p in param_counts]
    
    plt.scatter(train_times, mse_values, s=normalized_params, alpha=0.7)
    
    for i, arch in enumerate(architectures):
        plt.annotate(arch, (train_times[i], mse_values[i]), 
                   fontsize=9, ha='center')
    
    plt.title('Training Time vs. Error (bubble size = parameter count)')
    plt.xlabel('Training Time per Epoch (seconds)')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    complexity_path = os.path.join(output_dir, f"architecture_complexity_vs_performance_{timestamp}.png")
    plt.savefig(complexity_path, dpi=300)
    plt.close()
    saved_paths.append(complexity_path)
    
    print(f"Created {len(saved_paths)} visualization charts")
    logging.info(f"Architecture visualizations saved to {output_dir}")
    return saved_paths

def extended_train_model_with_tracking(model, train_loader, val_loader, test_loader,
                                     epochs=20, lr=1e-3, device='cpu',
                                     ticker='', architecture_name='', target_cols=[],
                                     # >> NEW: Accept features and analysis dict <<
                                     used_features: Optional[List[str]] = None,
                                     model_analysis_dict: Optional[Dict[str, Any]] = None):
    """
    Extended training function that tracks and logs performance metrics, including features/architecture.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to use ('cpu', 'cuda', 'mps')
        ticker: Stock ticker symbol (for logging)
        architecture_name: Name of model architecture (for logging)
        target_cols: Target columns being predicted
        used_features (Optional[List[str]]): List of feature names used.
        model_analysis_dict (Optional[Dict[str, Any]]): Dictionary with model architecture analysis.

    Returns:
        Tuple of (trained_model, history_dict, log_file_path)
    """
    log_dir = "performance_logs"
    logging.info(f"Initiating extended training with tracking for {ticker} - {architecture_name}")

    # Use track_performance for consistent behavior, passing new args
    log_path, history = track_performance(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader, # Pass test_loader for evaluation within track_performance
        epochs=epochs,
        ticker=ticker,
        architecture_name=architecture_name,
        target_cols=target_cols,
        save_dir=log_dir,
        verbose=True,  # Enable verbose console output during tracking
        lr=lr,         # Pass learning rate
        # >> Pass features and analysis down <<
        used_features=used_features,
        model_analysis_dict=model_analysis_dict
    )

    # The history dict is now returned directly by track_performance
    # No need to manually extract from the log file anymore

    logging.info(f"Extended training for {ticker} - {architecture_name} complete. Log: {log_path}")
    return model, history, log_path
