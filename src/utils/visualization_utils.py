import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch
import numpy as np

def save_and_display_results(model, history, analysis, ticker, target_cols, models_dir="models"):
    """
    Save the model and training plots. The model filename will be suffixed by the
    target columns and ticker.
    """
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    target_str = "-".join(target_cols)
    
    # Save model with filename indicating what it predicts and the ticker it was trained on
    model_save_path = f"{models_dir}/mixed_lstm_gru_model_target_{target_str}_trained_{ticker}_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label="Train Loss", linewidth=2)
    plt.plot(history['val_losses'], label="Validation Loss", linewidth=2)
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    plot_save_path = f"{models_dir}/training_plot_{target_str}_trained_{ticker}_{timestamp}.png"
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_save_path}")
    
    # Plot predicted vs actual values if available in history
    if 'y_true' in history and 'y_pred' in history:
        plot_predictions(history['y_true'], history['y_pred'], target_cols, ticker, models_dir, timestamp)
    
    print("\nModel Architecture Analysis:")
    print("-" * 50)
    print(f"Total parameters: {analysis['total_parameters']:,}")
    print(f"Trainable parameters: {analysis['trainable_parameters']:,}")
    print("\nLayer Shapes:")
    for layer_name, shapes in analysis['layer_shapes'].items():
        print(f"\n{layer_name}:")
        print(f"  Input shape: {shapes['input_shape']}")
        print(f"  Output shape: {shapes['output_shape']}")

def display_model_analysis(analysis):
    """Display the model architecture analysis in a formatted way."""
    print("\nNetwork Architecture Analysis:")
    print("-" * 50)
    print(f"Total parameters: {analysis['total_parameters']:,}")
    print(f"Trainable parameters: {analysis['trainable_parameters']:,}")
    print("\nLayer Shapes:")
    for layer_name, shapes in analysis['layer_shapes'].items():
        print(f"\n{layer_name}:")
        print(f"  Input shape: {shapes['input_shape']}")
        print(f"  Output shape: {shapes['output_shape']}")

def plot_predictions(y_true, y_pred, target_cols, ticker, output_dir="models", timestamp=None):
    """
    Plot predicted vs actual values for each target column.
    
    Args:
        y_true: Numpy array of true values
        y_pred: Numpy array of predicted values
        target_cols: List of target column names
        ticker: Stock ticker
        output_dir: Directory to save plots
        timestamp: Timestamp for filenames (optional)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%m%d%H%M%S")
    
    # Convert to numpy arrays if they're not already
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Ensure proper shapes
    if len(y_true_np.shape) == 1:
        y_true_np = y_true_np.reshape(-1, 1)
    if len(y_pred_np.shape) == 1:
        y_pred_np = y_pred_np.reshape(-1, 1)
    
    # Number of samples to plot (limit to avoid overcrowding)
    num_samples = min(250, len(y_true_np))
    indices = np.linspace(0, len(y_true_np)-1, num_samples, dtype=int)
    
    # Create a figure with subplots for each target column
    n_targets = y_true_np.shape[1]
    fig, axs = plt.subplots(n_targets, 1, figsize=(12, 5*n_targets), dpi=100)
    
    # Handle case with single target
    if n_targets == 1:
        axs = [axs]
    
    for i in range(n_targets):
        target_name = target_cols[i] if i < len(target_cols) else f"Target {i+1}"
        
        # Plot time series comparison
        axs[i].plot(indices, y_true_np[indices, i], 'b-', label=f'Actual {target_name}', linewidth=2)
        axs[i].plot(indices, y_pred_np[indices, i], 'r--', label=f'Predicted {target_name}', linewidth=2)
        axs[i].set_title(f'{ticker} - {target_name} Prediction vs Actual')
        axs[i].set_xlabel('Sample Index')
        axs[i].set_ylabel(f'{target_name} Value')
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    predictions_plot_path = os.path.join(output_dir, f"{ticker}_predictions_{timestamp}.png")
    plt.savefig(predictions_plot_path, dpi=300, bbox_inches='tight')
    print(f"Predictions plot saved to {predictions_plot_path}")
    
    # Create scatter plot for each target
    fig, axs = plt.subplots(1, n_targets, figsize=(7*n_targets, 6), dpi=100)
    
    # Handle case with single target
    if n_targets == 1:
        axs = [axs]
    
    for i in range(n_targets):
        target_name = target_cols[i] if i < len(target_cols) else f"Target {i+1}"
        
        # Scatter plot of predicted vs actual
        axs[i].scatter(y_true_np[:, i], y_pred_np[:, i], alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(np.max(y_true_np[:, i]), np.max(y_pred_np[:, i]))
        min_val = min(np.min(y_true_np[:, i]), np.min(y_pred_np[:, i]))
        axs[i].plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        axs[i].set_title(f'{ticker} - {target_name}')
        axs[i].set_xlabel(f'Actual {target_name}')
        axs[i].set_ylabel(f'Predicted {target_name}')
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_plot_path = os.path.join(output_dir, f"{ticker}_pred_scatter_{timestamp}.png")
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {scatter_plot_path}")

def plot_model_predictions(model, data_loader, target_cols, ticker, output_dir="models", device='cpu'):
    """
    Run model predictions on a data loader and plot predicted vs actual values.
    
    Args:
        model: PyTorch model to use for predictions
        data_loader: DataLoader with test data
        target_cols: List of target column names
        ticker: Stock ticker symbol
        output_dir: Directory to save plots
        device: Device to run model on ('cpu' or 'cuda')
    
    Returns:
        Tuple of numpy arrays (y_true, y_pred)
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for x_seq, y_val in data_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            y_pred = model(x_seq)
            all_y_true.extend(y_val.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())
    
    y_true_np = np.array(all_y_true)
    y_pred_np = np.array(all_y_pred)
    
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    plot_predictions(y_true_np, y_pred_np, target_cols, ticker, output_dir, timestamp)
    
    return y_true_np, y_pred_np