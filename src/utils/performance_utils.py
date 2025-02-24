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

from utils.model_utils import calculate_errors

def track_performance(model, train_loader, val_loader, test_loader, 
                     epochs: int, ticker: str, architecture_name: str,
                     target_cols: List[str], save_dir: str = "performance_logs") -> str:
    """
    Track and log comprehensive model performance metrics to a text file.
    
    Args:
        model: PyTorch model to evaluate
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        ticker: Stock ticker symbol
        architecture_name: Name of the model architecture
        target_cols: Target columns being predicted
        save_dir: Directory to save performance logs
        
    Returns:
        Path to the created log file
    """
    # Create log directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{architecture_name}_{ticker}_{timestamp}.txt"
    log_path = os.path.join(save_dir, log_filename)
    
    # Get device
    device = next(model.parameters()).device
    
    # Prepare log file
    with open(log_path, 'w') as f:
        # Write header information
        f.write(f"{'='*80}\n")
        f.write(f"MODEL PERFORMANCE REPORT\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Model Architecture: {architecture_name}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Target Columns: {', '.join(target_cols)}\n")
        start_time = datetime.datetime.now()
        f.write(f"Run Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model complexity metrics
        f.write(f"{'-'*80}\n")
        f.write(f"MODEL COMPLEXITY METRICS\n")
        f.write(f"{'-'*80}\n\n")
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        
        # Memory usage estimate (rough approximation)
        param_memory = total_params * 4 / (1024 * 1024)  # Size in MB assuming float32
        f.write(f"Estimated Model Size: {param_memory:.2f} MB\n\n")
        
        # Training metrics
        f.write(f"{'-'*80}\n")
        f.write(f"TRAINING METRICS\n")
        f.write(f"{'-'*80}\n\n")
        
        # Prepare for training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Initialize tracking variables
        train_losses = []
        val_losses = []
        epoch_times = []
        epoch_memory_usage = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        convergence_epoch = None
        
        # Training loop
        f.write(f"{'Epoch':^6}|{'Train Loss':^12}|{'Val Loss':^12}|{'Time (s)':^10}|{'Memory (MB)':^12}\n")
        f.write(f"{'-'*6}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*12}\n")
        
        for epoch in range(epochs):
            # Track epoch start time
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            for x_seq, y_true in train_loader:
                x_seq, y_true = x_seq.to(device), y_true.to(device)
                
                optimizer.zero_grad()
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average train loss
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_seq, y_true in val_loader:
                    x_seq, y_true = x_seq.to(device), y_true.to(device)
                    y_pred = model(x_seq)
                    val_loss += criterion(y_pred, y_true).item()
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Get current memory usage
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
            epoch_memory_usage.append(memory_usage)
            
            # Check for convergence (early stopping logic)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # If we have convergence and haven't recorded it yet
            if epochs_without_improvement >= 5 and convergence_epoch is None:
                convergence_epoch = epoch - 5  # When we last saw improvement
            
            # Write epoch results to log
            f.write(f"{epoch+1:^6}|{train_loss:^12.6f}|{val_loss:^12.6f}|{epoch_time:^10.2f}|{memory_usage:^12.2f}\n")
        
        # Calculate average training time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        f.write(f"\nAverage Time per Epoch: {avg_epoch_time:.2f} seconds\n")
        
        # Record convergence information
        if convergence_epoch is not None:
            f.write(f"Convergence at Epoch: {convergence_epoch + 1} (no improvement for 5 epochs after)\n")
        else:
            f.write(f"No clear convergence detected within {epochs} epochs\n")
        
        # Model evaluation on test set
        f.write(f"\n{'-'*80}\n")
        f.write(f"TEST SET EVALUATION\n")
        f.write(f"{'-'*80}\n\n")
        
        # Start timing for inference
        inference_start = time.time()
        
        # Test evaluation
        model.eval()
        test_loss = 0.0
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for x_seq, y_true in test_loader:
                x_seq, y_true = x_seq.to(device), y_true.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_true)
                test_loss += loss.item()
                
                all_y_true.extend(y_true.cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy())
        
        # Calculate test loss
        test_loss /= len(test_loader)
        
        # Convert to numpy arrays
        y_true_np = np.array(all_y_true)
        y_pred_np = np.array(all_y_pred)
        
        # Calculate inference time
        total_samples = len(test_loader.dataset)
        inference_time = time.time() - inference_start
        inference_time_per_sample = inference_time / total_samples
        
        # Calculate standard error metrics using the existing function
        errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
        
        # Calculate directional accuracy
        directional_accuracy = calculate_directional_accuracy(y_true_np, y_pred_np)
        
        # Calculate maximum error
        max_error = calculate_max_error(y_true_np, y_pred_np)
        
        # Write error metrics
        f.write(f"MSE: {errors['mse']:.6f}\n")
        f.write(f"RMSE: {errors['rmse']:.6f}\n")
        f.write(f"MAE: {errors['mae']:.6f}\n")
        f.write(f"MAPE: {errors['mape']:.2f}%\n")
        f.write(f"Directional Accuracy: {directional_accuracy:.2f}%\n")
        f.write(f"Maximum Error: {max_error:.6f}\n\n")
        
        # Write inference performance
        f.write(f"Total Inference Time: {inference_time:.4f} seconds\n")
        f.write(f"Inference Time per Sample: {inference_time_per_sample*1000:.4f} ms\n\n")
        
        # Write closing timestamp
        end_time = datetime.datetime.now()
        f.write(f"Run Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write(f"Total Duration: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}\n")
        
    logging.info(f"Performance metrics saved to {log_path}")
    return log_path

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
    
    for model_config in models:
        model_name = model_config['name']
        model = model_config['model']
        
        print(f"\nEvaluating {model_name} architecture...")
        
        log_path = track_performance(
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            test_loader=data_loaders['test'],
            epochs=epochs,
            ticker=ticker,
            architecture_name=model_name,
            target_cols=target_cols,
            save_dir=save_dir
        )
        
        log_paths.append(log_path)
        print(f"Performance log saved to: {log_path}")
    
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
    
    logging.info(f"Architecture visualizations saved to {output_dir}")
    return saved_paths

def extended_train_model_with_tracking(model, train_loader, val_loader, test_loader, 
                                     epochs=20, lr=1e-3, device='cpu',
                                     ticker='', architecture_name='', target_cols=[]):
    """
    Extended training function that also tracks and logs performance metrics.
    
    This combines functionality from model_utils.train_model with performance tracking.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to use ('cpu' or 'cuda')
        ticker: Stock ticker symbol (for logging)
        architecture_name: Name of model architecture (for logging)
        target_cols: Target columns being predicted
        
    Returns:
        Tuple of (trained_model, history, log_path)
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Create log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "performance_logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_filename = f"{architecture_name}_{ticker}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)
    
    with open(log_path, 'w') as f:
        # Write header information
        f.write(f"{'='*80}\n")
        f.write(f"MODEL PERFORMANCE REPORT\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Model Architecture: {architecture_name}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Target Columns: {', '.join(target_cols)}\n")
        start_time = datetime.datetime.now()
        f.write(f"Run Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model complexity metrics
        f.write(f"{'-'*80}\n")
        f.write(f"MODEL COMPLEXITY METRICS\n")
        f.write(f"{'-'*80}\n\n")
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        
        # Memory usage estimate
        param_memory = total_params * 4 / (1024 * 1024)
        f.write(f"Estimated Model Size: {param_memory:.2f} MB\n\n")
        
        # Training metrics
        f.write(f"{'-'*80}\n")
        f.write(f"TRAINING METRICS\n")
        f.write(f"{'-'*80}\n\n")
        
        # Write table header
        f.write(f"{'Epoch':^6}|{'Train Loss':^12}|{'Val Loss':^12}|{'Time (s)':^10}|{'Memory (MB)':^12}|{'LR':^10}\n")
        f.write(f"{'-'*6}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*10}\n")
    
    # Early stopping setup
    early_stopping = torch.utils.data.DataLoader
    early_stopping_counter = 0
    best_val_loss = float('inf')
    convergence_epoch = None
    
    model.to(device)
    train_losses = []
    val_losses = []
    epoch_times = []
    
    print("\nStarting training with performance tracking...")
    for epoch in range(epochs):
        # Track epoch start time
        epoch_start = time.time()
        
        # Training
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
        
        # Validation
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
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Get current memory usage
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 5 and convergence_epoch is None:
                convergence_epoch = epoch - 5
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
        # Write to log file
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1:^6}|{avg_train_loss:^12.6f}|{avg_val_loss:^12.6f}|{epoch_time:^10.2f}|{memory_usage:^12.2f}|{current_lr:^10.2e}\n")
        
        # Early stopping check
        if early_stopping_counter >= 5:
            print("\nEarly stopping triggered")
            break
    
    # Finish training section in log
    with open(log_path, 'a') as f:
        # Calculate average training time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        f.write(f"\nAverage Time per Epoch: {avg_epoch_time:.2f} seconds\n")
        
        # Record convergence information
        if convergence_epoch is not None:
            f.write(f"Convergence at Epoch: {convergence_epoch + 1} (no improvement for 5 epochs after)\n")
        else:
            f.write(f"No clear convergence detected within {epochs} epochs\n")
        
        # Model evaluation on test set
        f.write(f"\n{'-'*80}\n")
        f.write(f"TEST SET EVALUATION\n")
        f.write(f"{'-'*80}\n\n")
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    all_y_true = []
    all_y_pred = []
    
    # Start timing for inference
    inference_start = time.time()
    
    with torch.no_grad():
        for x_seq, y_true in test_loader:
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            test_loss += loss.item()
            
            all_y_true.extend(y_true.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())
    
    # Calculate test loss
    test_loss /= len(test_loader)
    
    # Calculate inference time
    total_samples = len(test_loader.dataset)
    inference_time = time.time() - inference_start
    inference_time_per_sample = inference_time / total_samples
    
    # Calculate standard error metrics
    errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
    
    # Calculate additional metrics
    y_true_np = np.array(all_y_true)
    y_pred_np = np.array(all_y_pred)
    directional_accuracy = calculate_directional_accuracy(y_true_np, y_pred_np)
    max_error = calculate_max_error(y_true_np, y_pred_np)
    
    # Write to log file
    with open(log_path, 'a') as f:
        f.write(f"MSE: {errors['mse']:.6f}\n")
        f.write(f"RMSE: {errors['rmse']:.6f}\n")
        f.write(f"MAE: {errors['mae']:.6f}\n")
        f.write(f"MAPE: {errors['mape']:.2f}%\n")
        f.write(f"Directional Accuracy: {directional_accuracy:.2f}%\n")
        f.write(f"Maximum Error: {max_error:.6f}\n\n")
        
        # Write inference performance
        f.write(f"Total Inference Time: {inference_time:.4f} seconds\n")
        f.write(f"Inference Time per Sample: {inference_time_per_sample*1000:.4f} ms\n\n")
        
        # Write closing timestamp
        end_time = datetime.datetime.now()
        f.write(f"Run Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write(f"Total Duration: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}\n")
    
    # Print test results to console
    print("\nTest Set Metrics:")
    print("-" * 50)
    print(f"MSE: {errors['mse']:.6f}")
    print(f"RMSE: {errors['rmse']:.6f}")
    print(f"MAE: {errors['mae']:.6f}")
    print(f"MAPE: {errors['mape']:.2f}%")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    print(f"Maximum Error: {max_error:.6f}")
    print(f"Inference Time per Sample: {inference_time_per_sample*1000:.4f} ms")
    
    # Prepare history dictionary
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    logging.info(f"Performance metrics saved to {log_path}")
    return model, history, log_path

def add_architecture_benchmark_to_menu(menu_utils_module):
    """
    Extends the menu_utils module with an architecture benchmarking option.
    
    Args:
        menu_utils_module: The imported menu_utils module
    """
    # Store the original display_menu function
    original_display_menu = menu_utils_module.display_menu
    
    # Define the extended menu function
    def extended_display_menu():
        """Display the extended main menu with architecture benchmarking."""
        print("\nOption Trading Model - Main Menu")
        print("-" * 50)
        print("1. Train new model")
        print("2. Run existing model")
        print("3. Analyze network architecture")
        print("4. Benchmark multiple architectures")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ")
                if choice in ['1', '2', '3', '4', '5']:
                    return int(choice)
                print("Invalid choice. Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Replace the original function with the extended one
    menu_utils_module.display_menu = extended_display_menu