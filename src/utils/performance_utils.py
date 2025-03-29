import torch
import numpy as np
import time
import os
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


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil library not found. Memory tracking for CPU/MPS will be unavailable.") # Inform user

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
                     used_features: Optional[List[str]] = None,
                     model_analysis_dict: Optional[Dict[str, Any]] = None,
                     # --- Add feature flags ---
                     include_greeks: Optional[bool] = None,
                     include_rolling: Optional[bool] = None,
                     include_cyclical: Optional[bool] = None,
                     # --- End feature flags ---
                     save_dir: str = "performance_logs",
                     verbose: bool = True,
                     lr: float = 0.001) -> Tuple[str, Dict]:
    """Track performance, now logging feature flag usage."""
    log_dir_path = Path(save_dir); plots_dir_path = log_dir_path / 'plots'
    log_dir_path.mkdir(exist_ok=True); plots_dir_path.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{architecture_name}_{ticker}_{timestamp}.txt"; log_path = log_dir_path / log_filename
    logging.info(f"Performance log: {log_path}")

    start_run_time = datetime.datetime.now()

    try: device = next(model.parameters()).device
    except StopIteration: device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    history = {'train_losses': [], 'val_losses': [], 'test_metrics': {}, 'epoch_times': []}

    with open(log_path, 'w') as f:
        # --- Header ---
        f.write(f"{'='*80}\nMODEL PERFORMANCE REPORT\n{'='*80}\n\n")
        f.write(f"Model Architecture: {architecture_name}\nTicker: {ticker}\n")
        f.write(f"Target Columns: {', '.join(target_cols)}\n")
        f.write(f"Run Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\nLearning Rate: {lr}\nMax Epochs: {epochs}\n")
        # Log feature flags
        f.write(f"Include Greeks: {include_greeks}\n")
        f.write(f"Include Rolling: {include_rolling}\n")
        f.write(f"Include Cyclical Dates: {include_cyclical}\n\n")

        # --- Log Features Used ---
        if used_features:
             f.write(f"{'-'*80}\nFEATURES USED ({len(used_features)})\n{'-'*80}\n")
             # ... (logic to write features remains the same) ...
             feature_lines = []; current_line = ""
             for feature in used_features:
                  if not current_line: current_line = feature
                  elif len(current_line) + len(feature) + 2 < 100: current_line += f", {feature}"
                  else: feature_lines.append(current_line); current_line = feature
             if current_line: feature_lines.append(current_line)
             f.write("\n".join(feature_lines) + "\n\n")
        else: f.write("Features used: Not Provided\n\n")

        # --- Log Model Architecture (remains the same) ---
        if model_analysis_dict:
            # ... (logic to write architecture details remains the same) ...
            f.write(f"{'-'*80}\nMODEL ARCHITECTURE DETAILS\n{'-'*80}\n")
            total_params = model_analysis_dict.get('total_parameters', 'N/A')
            trainable_params = model_analysis_dict.get('trainable_parameters', 'N/A')
            total_params_str = f"{total_params:,}" if isinstance(total_params, int) else str(total_params)
            trainable_params_str = f"{trainable_params:,}" if isinstance(trainable_params, int) else str(trainable_params)
            f.write(f"Total Parameters: {total_params_str}\nTrainable Parameters: {trainable_params_str}\n\n")
            f.write("Layer Shapes (Input -> Output):\n")
            layer_shapes = model_analysis_dict.get('layer_shapes', {})
            if isinstance(layer_shapes, dict) and layer_shapes:
                 for layer_name, shapes in layer_shapes.items():
                      if isinstance(shapes, dict):
                           in_shape = shapes.get('input_shape', 'N/A'); out_shape = shapes.get('output_shape', 'N/A')
                           f.write(f"  {layer_name}: Input: {in_shape} -> Output: {out_shape}\n")
                      else: f.write(f"  {layer_name}: Shape details not available ({shapes})\n")
            else: f.write("  Layer shape details not available.\n")
            f.write("\n")
        else: f.write("Model Analysis: Not Provided\n\n")

        # --- Training Header ---
        f.write(f"{'-'*80}\nTRAINING METRICS\n{'-'*80}\n\n")
        f.write(f"{'Epoch':^6}|{'Train Loss':^12}|{'Val Loss':^12}|{'LR':^10}|{'Time (s)':^10}|{'Memory (MB)':^12}\n")
        f.write(f"{'-'*6}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*12}\n")

    # --- Training Loop (remains the same logic) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
    early_stopping = EarlyStopping(patience=5, min_delta=1e-5)
    total_train_time = 0.0; actual_epochs_run = 0

    if verbose: print(f"\n{'-' * 80}\n{architecture_name} - TRAINING PHASE ({ticker})\n{'-' * 80}") # ... (verbose print setup) ...

    for epoch in range(epochs):
        # Start measuring epoch time here, at the beginning of the epoch
        epoch_start_time = time.time()
        
        # --- Training phase ---
        model.train(); total_train_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch", disable=not verbose, leave=False)
        for x_seq, y_val in train_iter:
             x_seq, y_val = x_seq.to(device), y_val.to(device); optimizer.zero_grad()
             y_pred = model(x_seq); loss = criterion(y_pred, y_val)
             if torch.isnan(loss): logging.warning(f"NaN loss at train epoch {epoch+1}"); break
             loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
             total_train_loss += loss.item(); train_iter.set_postfix({'loss': f'{loss.item():.6f}'})
        if 'loss' in locals() and torch.isnan(loss): break # Exit epoch loop if NaN
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        history['train_losses'].append(avg_train_loss)

        # --- Validation phase ---
        model.eval(); total_val_loss = 0.0
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", unit="batch", disable=not verbose, leave=False)
        with torch.no_grad():
            for x_seq, y_val in val_iter:
                x_seq, y_val = x_seq.to(device), y_val.to(device)
                y_pred = model(x_seq); loss = criterion(y_pred, y_val)
                total_val_loss += loss.item(); val_iter.set_postfix({'val_loss': f'{loss.item():.6f}'})
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        history['val_losses'].append(avg_val_loss)

        # --- Scheduler, Early Stopping, Logging ---
        scheduler.step(avg_val_loss); early_stopping(avg_val_loss)
        epoch_time = time.time() - epoch_start_time  # Calculate total epoch time
        history['epoch_times'].append(epoch_time); total_train_time += epoch_time
        memory_allocated = np.nan
        try:
             if str(device).startswith('cuda'): memory_allocated = torch.cuda.memory_allocated(device) / (1024*1024)
             elif str(device).startswith('mps') and PSUTIL_AVAILABLE: memory_allocated = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
             elif PSUTIL_AVAILABLE: memory_allocated = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
        except Exception as mem_err: logging.warning(f"Mem fetch err: {mem_err}")

        with open(log_path, 'a') as f:
             current_lr = optimizer.param_groups[0]['lr']; mem_str = f"{memory_allocated:.2f}" if not np.isnan(memory_allocated) else "N/A"
             f.write(f"{epoch+1:^6}|{avg_train_loss:^12.6f}|{avg_val_loss:^12.6f}|{current_lr:^10.2e}|{epoch_time:^10.2f}|{mem_str:^12}\n")
        if verbose: print(f"\nEpoch [{epoch+1}/{epochs}] Summary: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e} | Time: {epoch_time:.2f}s | Mem: {mem_str} MB")
        actual_epochs_run += 1 # Increment counter inside loop
        if early_stopping.early_stop: print(f"\nEarly stopping @ epoch {epoch+1}"); break


    # --- Training Summary (write to log) ---
    with open(log_path, 'a') as f:
        avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times']) if history['epoch_times'] else 0
        f.write(f"\n--- Training Summary ---\n")
        f.write(f"Training stopped after epoch {actual_epochs_run}\n") # Use actual count
        f.write(f"Total Training Time: {total_train_time:.2f} seconds\n")
        f.write(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds\n")
        best_loss_str = f"{early_stopping.best_loss:.6f}" if early_stopping.best_loss is not None else "N/A"
        f.write(f"Best Validation Loss: {best_loss_str}\n")

    # --- Test Set Evaluation (remains the same logic) ---
    # ... (test loop, metric calculation, logging to file, optional plotting) ...
    with open(log_path, 'a') as f: f.write(f"\n{'-'*80}\nTEST SET EVALUATION\n{'-'*80}\n\n")
    if verbose: print(f"\n{'-' * 80}\n{architecture_name} - TESTING PHASE ({ticker})\n{'-' * 80}")
    inference_start_time = time.time(); model.eval(); test_loss = 0.0
    all_y_true_test = []; all_y_pred_test = []
    with torch.no_grad():
         test_iter = tqdm(test_loader, desc="Testing", unit="batch", disable=not verbose, leave=False)
         for x_seq, y_true in test_iter:
              x_seq, y_true = x_seq.to(device), y_true.to(device)
              y_pred = model(x_seq); loss = criterion(y_pred, y_true); test_loss += loss.item()
              all_y_true_test.extend(y_true.cpu().numpy()); all_y_pred_test.extend(y_pred.cpu().numpy())
    avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    inference_time = time.time() - inference_start_time; total_test_samples = len(all_y_true_test)
    inf_time_per_sample = (inference_time / total_test_samples * 1000) if total_test_samples > 0 else 0
    y_true_np = np.array(all_y_true_test); y_pred_np = np.array(all_y_pred_test)
    if y_true_np.size > 0 and y_pred_np.size > 0:
         test_errors = calculate_errors(torch.tensor(y_true_np), torch.tensor(y_pred_np))
         directional_accuracy = calculate_directional_accuracy(y_true_np, y_pred_np)
         max_error = calculate_max_error(y_true_np, y_pred_np)
         history['test_metrics'] = {**test_errors, 'directional_accuracy': directional_accuracy, 'max_error': max_error}
    else: history['test_metrics'] = {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'directional_accuracy': np.nan, 'max_error': np.nan}
    with open(log_path, 'a') as f: # Log test metrics
        f.write(f"Test Loss (MSE): {avg_test_loss:.6f}\n"); f.write(f"Test RMSE: {history['test_metrics'].get('rmse', np.nan):.6f}\n")
        f.write(f"Test MAE: {history['test_metrics'].get('mae', np.nan):.6f}\n"); f.write(f"Test MAPE: {history['test_metrics'].get('mape', np.nan):.2f}%\n")
        f.write(f"Test Directional Accuracy: {history['test_metrics'].get('directional_accuracy', np.nan):.2f}%\n")
        f.write(f"Test Maximum Error: {history['test_metrics'].get('max_error', np.nan):.6f}\n\n")
        f.write(f"Total Inference Time: {inference_time:.4f}s ({total_test_samples} samples)\n")
        f.write(f"Inference Time per Sample: {inf_time_per_sample:.4f} ms\n\n")
    history['y_true'] = all_y_true_test; history['y_pred'] = all_y_pred_test

    # --- Plotting (remains the same logic, uses plots_dir_path) ---
    plot_files = None
    if total_test_samples > 0:
         try:
             plot_timestamp = timestamp # Reuse timestamp
             if verbose: print(f"\nGenerating prediction plots...")
             # Ensure plot_predictions is imported/available
             plot_files = plot_predictions(
                  y_true=y_true_np, y_pred=y_pred_np, target_cols=target_cols, ticker=ticker,
                  output_dir=str(plots_dir_path), # Save plots in the subdir
                  timestamp=plot_timestamp,
                  rmse=history['test_metrics'].get('rmse'), mae=history['test_metrics'].get('mae'),
                  plot_suffix=f"_{architecture_name}"
             )
             with open(log_path, 'a') as f: # Log plot paths
                  f.write(f"Prediction Plots Saved To: {plots_dir_path}\n");
                  if plot_files:
                       for pt, fp in plot_files.items(): f.write(f"  {pt.replace('_',' ').title()}: {Path(fp).name}\n")
                  f.write("\n")
             if verbose and plot_files: print(f"Prediction plots saved to {plots_dir_path}")
         except Exception as plot_err: logging.error(f"Plot gen failed: {plot_err}");
    else:
         with open(log_path, 'a') as f: f.write("No test samples for plots.\n\n")

    # --- Closing (remains the same logic) ---
    end_run_time = datetime.datetime.now(); duration = end_run_time - start_run_time
    with open(log_path, 'a') as f:
        f.write(f"Run Completed: {end_run_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {str(duration).split('.')[0]}\n")
    # ... (verbose print summary remains the same) ...
    if verbose:
         print(f"\n{'-'*80}\nTEST SET FINAL METRICS ({ticker} - {architecture_name})\n{'-'*50}")
         print(f"Test Loss (MSE): {avg_test_loss:.6f}"); print(f"Test RMSE: {history['test_metrics'].get('rmse', np.nan):.6f}")
         print(f"Test MAE: {history['test_metrics'].get('mae', np.nan):.6f}"); print(f"Test MAPE: {history['test_metrics'].get('mape', np.nan):.2f}%")
         print(f"Directional Accuracy: {history['test_metrics'].get('directional_accuracy', np.nan):.2f}%"); print(f"Maximum Error: {history['test_metrics'].get('max_error', np.nan):.6f}")
         print(f"Inference Time/Sample: {inf_time_per_sample:.4f} ms"); print(f"\nTotal Run Duration: {str(duration).split('.')[0]}")
         print(f"Log saved to: {log_path}"); print(f"Plots saved in: {plots_dir_path}" if plot_files else "No plots generated.")
         print(f"{'-'*80}\n")

    logging.info(f"Performance metrics saved to {log_path}")
    return str(log_path), history

def benchmark_architectures(models: List[Dict[str, Any]],
                          data_loaders: Dict[str, Any],
                          epochs: int,
                          ticker: str,
                          target_cols: List[str],
                          save_dir: str = "performance_logs",
                          device: str = 'cpu', # Add device and lr
                          lr: float = 1e-3,
                          # --- Add feature flags ---
                          include_greeks: Optional[bool] = None,
                          include_rolling: Optional[bool] = None,
                          include_cyclical: Optional[bool] = None
                          ) -> List[str]:
    """Compare models, passing feature flags to track_performance."""
    log_paths = []
    total_models = len(models)
    logging.info(f"Starting benchmark for {total_models} architectures on {ticker}.")
    # ... (print benchmark header) ...
    print(f"\n{'='*80}\nBENCHMARK COMPARISON: {total_models} ARCHITECTURES ON {ticker}\n{'='*80}")
    print(f"Features: Greeks={include_greeks}, Rolling={include_rolling}, Cyclical={include_cyclical}")

    benchmark_start = time.time()
    for i, model_config in enumerate(models, 1):
        model_name = model_config['name']
        model = model_config['model']
        model_analysis = model_config.get('analysis') # Get analysis dict if provided
        model = model.to(device) # Ensure model is on correct device

        print(f"\n{'='*80}\n[{i}/{total_models}] BENCHMARKING: {model_name}\n{'='*80}")
        model_start = time.time()

        # Call track_performance with all flags
        log_path, _ = track_performance( # History dict isn't needed here
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            test_loader=data_loaders['test'],
            epochs=epochs,
            ticker=ticker,
            architecture_name=model_name,
            target_cols=target_cols,
            save_dir=save_dir,
            verbose=True,
            lr=lr,
            used_features=data_loaders['train'].dataset.dataset.feature_cols, # Get features from underlying dataset
            model_analysis_dict=model_analysis,
            # --- Pass flags ---
            include_greeks=include_greeks,
            include_rolling=include_rolling,
            include_cyclical=include_cyclical
        )
        log_paths.append(log_path)
        model_time = time.time() - model_start; hours, rem = divmod(model_time,3600); mins, secs = divmod(rem,60)
        print(f"\nCompleted {model_name} in {int(hours):02d}:{int(mins):02d}:{secs:.2f}. Log: {log_path}")

    benchmark_time = time.time() - benchmark_start; hours, rem = divmod(benchmark_time,3600); mins, secs = divmod(rem,60)
    print(f"\n{'='*80}\nBENCHMARK COMPLETE\n{'='*80}\nTotal time: {int(hours):02d}:{int(mins):02d}:{secs:.2f}. Logs in: {save_dir}")
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

def extended_train_model_with_tracking(
    model, train_loader, val_loader, test_loader,
    epochs=20, lr=1e-3, device='cpu', # Keep device here, it's passed TO this function
    ticker='', architecture_name='', target_cols=[],
    used_features: Optional[List[str]] = None,
    model_analysis_dict: Optional[Dict[str, Any]] = None,
    performance_logs_dir: str = "performance_logs",
    include_greeks: Optional[bool] = None,
    include_rolling: Optional[bool] = None,
    include_cyclical: Optional[bool] = None
    ):
    """
    Extended training function that calls track_performance.
    Removed the 'device' argument from the track_performance call.
    """
    logging.info(f"Initiating extended training with tracking for {ticker} - {architecture_name}")
    logging.info(f"Performance logs will be saved to: {performance_logs_dir}")

    log_path, history = track_performance(
        model=model, # Pass model (track_performance gets device from this)
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        ticker=ticker,
        architecture_name=architecture_name,
        target_cols=target_cols,
        save_dir=performance_logs_dir, # Pass log directory
        verbose=True,
        lr=lr,
        used_features=used_features,
        model_analysis_dict=model_analysis_dict,
        include_greeks=include_greeks,
        include_rolling=include_rolling,
        include_cyclical=include_cyclical
    )

    logging.info(f"Extended training complete. Log: {log_path}")
    return model, history, log_path
