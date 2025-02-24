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
                     target_cols: List[str], save_dir: str = "performance_logs",
                     verbose: bool = True) -> str:
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
        verbose: Whether to print detailed progress to terminal
        
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
        
        if verbose:
            print(f"\n{'-' * 80}")
            print(f"{architecture_name} - TRAINING PHASE")
            print(f"{'-' * 80}")
            print(f"Parameters: {trainable_params:,}")
            print(f"Device: {device}")
            print(f"Training samples: {len(train_loader.dataset):,}")
            print(f"Validation samples: {len(val_loader.dataset):,}")
            print(f"Batches per epoch: {len(train_loader):,}")
            print(f"{'-' * 50}")
        
        for epoch in range(epochs):
            # Track epoch start time
            epoch_start = time.time()
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"{'-' * 20}")
            
            # Training
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            # Progress bar
            if verbose:
                train_iter = tqdm(train_loader, desc="Training", unit="batch")
            else:
                train_iter = train_loader
                
            for x_seq, y_true in train_iter:
                x_seq, y_true = x_seq.to(device), y_true.to(device)
                
                optimizer.zero_grad()
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                # Update progress bar with current loss
                if verbose and batch_count % 10 == 0:
                    train_iter.set_postfix({"loss": f"{loss.item():.6f}"})
            
            # Calculate average train loss
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            # Progress bar
            if verbose:
                val_iter = tqdm(val_loader, desc="Validation", unit="batch")
            else:
                val_iter = val_loader
                
            with torch.no_grad():
                for x_seq, y_true in val_iter:
                    x_seq, y_true = x_seq.to(device), y_true.to(device)
                    y_pred = model(x_seq)
                    loss = criterion(y_pred, y_true)
                    val_loss += loss.item()
            
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
                improvement = "✓ (improved)"
            else:
                epochs_without_improvement += 1
                improvement = f"(no improvement: {epochs_without_improvement})"
            
            # If we have convergence and haven't recorded it yet
            if epochs_without_improvement >= 5 and convergence_epoch is None:
                convergence_epoch = epoch - 5  # When we last saw improvement
            
            # Write epoch results to log
            f.write(f"{epoch+1:^6}|{train_loss:^12.6f}|{val_loss:^12.6f}|{epoch_time:^10.2f}|{memory_usage:^12.2f}\n")
            
            # Print status to terminal if verbose
            if verbose:
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss: {val_loss:.6f} {improvement}")
                print(f"Time: {epoch_time:.2f}s")
                print(f"Memory: {memory_usage:.1f} MB")
                
            # Early stopping check
            if epochs_without_improvement >= 5:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
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
        
        if verbose:
            print(f"\n{'-' * 80}")
            print(f"{architecture_name} - TESTING PHASE")
            print(f"{'-' * 80}")
            print(f"Test samples: {len(test_loader.dataset):,}")
        
        # Start timing for inference
        inference_start = time.time()
        
        # Test evaluation
        model.eval()
        test_loss = 0.0
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            # Progress bar for test set
            if verbose:
                test_iter = tqdm(test_loader, desc="Testing", unit="batch")
            else:
                test_iter = test_loader
                
            for x_seq, y_true in test_iter:
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
        
        # Calculate error metrics
        mse = np.mean((y_true_np - y_pred_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true_np - y_pred_np))
        
        # Calculate MAPE with handling for zero values
        epsilon = 1e-10  # Small value to avoid division by zero
        abs_percentage_error = np.abs((y_true_np - y_pred_np) / np.maximum(np.abs(y_true_np), epsilon)) * 100
        mape = np.mean(abs_percentage_error)
        
        # Calculate directional accuracy if we have enough data
        if len(y_true_np) > 1:
            directional_accuracy = calculate_directional_accuracy(y_true_np, y_pred_np)
        else:
            directional_accuracy = None
        
        # Calculate maximum error
        max_error = calculate_max_error(y_true_np, y_pred_np)
        
        # Write error metrics
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        
        if directional_accuracy is not None:
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
        
        if verbose:
            print(f"\nTest Results:")
            print(f"{'-' * 30}")
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"MAPE: {mape:.2f}%")
            if directional_accuracy is not None:
                print(f"Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"Maximum Error: {max_error:.6f}")
            print(f"Inference Time per Sample: {inference_time_per_sample*1000:.4f} ms")
            print(f"\nTotal Duration: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
            print(f"Log saved to: {log_path}")
            print(f"{'-' * 80}\n")
    
    logging.info(f"Performance metrics saved to {log_path}")
    return log_path

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
        
        # Clear separator between models
        print(f"\n{'=' * 80}")
        print(f"[{i}/{total_models}] BENCHMARKING: {model_name}")
        print(f"{'=' * 80}")
        
        # Display parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {param_count:,}")
        
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
    # Create log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "performance_logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_filename = f"{architecture_name}_{ticker}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)
    
    # Use track_performance for consistent behavior
    log_path = track_performance(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        ticker=ticker,
        architecture_name=architecture_name,
        target_cols=target_cols,
        save_dir=log_dir,
        verbose=True  # Enable verbose output
    )
    
    # Get history for visualization
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            # Extract train and validation losses
            train_loss_lines = [line for line in content.split('\n') if '|' in line and not line.startswith('-')]
            train_losses = []
            val_losses = []
            
            for line in train_loss_lines:
                if line[0].isdigit():  # Skip header line
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            train_loss = float(parts[1].strip())
                            val_loss = float(parts[2].strip())
                            train_losses.append(train_loss)
                            val_losses.append(val_loss)
                        except ValueError:
                            pass
            
            history = {
                'train_losses': train_losses,
                'val_losses': val_losses
            }
    except Exception as e:
        logging.error(f"Error extracting history from log: {str(e)}")
        history = {'train_losses': [], 'val_losses': []}
    
    return model, history, log_path