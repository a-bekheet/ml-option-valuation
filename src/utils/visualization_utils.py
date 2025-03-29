import logging
from typing import Optional
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

def plot_predictions(y_true, y_pred, target_cols, ticker, output_dir="models", timestamp=None,
                     rmse: Optional[float] = None, mae: Optional[float] = None, n_samples_plot: int = 250):
    """
    Plot predicted vs actual values for each target column using original scale
    and annotate with provided error metrics.

    Args:
        y_true: Numpy array of true values (original scale).
        y_pred: Numpy array of predicted values (original scale).
        target_cols: List of target column names.
        ticker: Stock ticker symbol.
        output_dir: Directory to save plots.
        timestamp: Timestamp for filenames (optional).
        rmse (Optional[float]): Root Mean Squared Error (original scale) for annotation.
        mae (Optional[float]): Mean Absolute Error (original scale) for annotation.
        n_samples_plot (int): Max number of samples for time series plot.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%m%d%H%M%S")

    # Ensure inputs are numpy arrays
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Ensure proper shapes (samples x targets)
    if len(y_true_np.shape) == 1: y_true_np = y_true_np.reshape(-1, 1)
    if len(y_pred_np.shape) == 1: y_pred_np = y_pred_np.reshape(-1, 1)

    num_targets = y_true_np.shape[1]
    if len(target_cols) != num_targets:
         logging.warning(f"Mismatch between target_cols ({len(target_cols)}) and data shape ({num_targets}). Using generic target names.")
         target_cols = [f"Target {i+1}" for i in range(num_targets)]

    # --- Time Series Plot ---
    num_samples = min(n_samples_plot, len(y_true_np))
    indices = np.linspace(0, len(y_true_np)-1, num_samples, dtype=int) if len(y_true_np) > 0 else []

    if len(indices) > 0: # Only plot if there's data
        fig_ts, axs_ts = plt.subplots(num_targets, 1, figsize=(12, 5 * num_targets), dpi=100, squeeze=False) # Ensure axs is 2D

        for i in range(num_targets):
            target_name = target_cols[i]
            axs_ts[i, 0].plot(indices, y_true_np[indices, i], 'b-', label=f'Actual {target_name}', linewidth=1.5)
            axs_ts[i, 0].plot(indices, y_pred_np[indices, i], 'r--', label=f'Predicted {target_name}', linewidth=1.5)
            axs_ts[i, 0].set_title(f'{ticker} - {target_name} Prediction vs Actual (Sample)')
            axs_ts[i, 0].set_xlabel('Sample Index')
            axs_ts[i, 0].set_ylabel(f'{target_name} Value ($)') # Indicate original scale
            axs_ts[i, 0].legend()
            axs_ts[i, 0].grid(True, alpha=0.3)

        plt.tight_layout()
        ts_plot_path = os.path.join(output_dir, f"{ticker}_predictions_ts_{timestamp}.png") # Changed filename slightly
        plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to {ts_plot_path}")
        plt.close(fig_ts) # Close the figure
    else:
         logging.warning("Skipping time series plot due to insufficient data.")
         ts_plot_path = None


    # --- Scatter Plot ---
    if len(y_true_np) > 0: # Only plot if there's data
        fig_sc, axs_sc = plt.subplots(1, num_targets, figsize=(7 * num_targets, 6), dpi=100, squeeze=False) # Ensure axs is 2D

        for i in range(num_targets):
            target_name = target_cols[i]
            axs_sc[0, i].scatter(y_true_np[:, i], y_pred_np[:, i], alpha=0.4, s=10) # Adjust alpha/size

            # Add perfect prediction line (y=x)
            min_val = min(np.min(y_true_np[:, i]), np.min(y_pred_np[:, i]))
            max_val = max(np.max(y_true_np[:, i]), np.max(y_pred_np[:, i]))
            axs_sc[0, i].plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction', linewidth=1)

            axs_sc[0, i].set_title(f'{ticker} - {target_name}')
            axs_sc[0, i].set_xlabel(f'Actual {target_name} ($)') # Indicate original scale
            axs_sc[0, i].set_ylabel(f'Predicted {target_name} ($)') # Indicate original scale
            axs_sc[0, i].grid(True, alpha=0.3)

            # --- Add Metric Annotations ---
            annotation_text = []
            if rmse is not None: annotation_text.append(f'RMSE: {rmse:.4f}')
            if mae is not None: annotation_text.append(f'MAE:  {mae:.4f}')
            # You could add other metrics here if passed

            if annotation_text:
                 # Place text in upper left corner
                 axs_sc[0, i].text(0.05, 0.95, "\n".join(annotation_text),
                                   transform=axs_sc[0, i].transAxes, # Use axes coordinates
                                   fontsize=9, verticalalignment='top',
                                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            # --- End Annotations ---

            axs_sc[0, i].legend(loc='lower right') # Adjust legend position maybe

        plt.tight_layout()
        scatter_plot_path = os.path.join(output_dir, f"{ticker}_pred_scatter_{timestamp}.png")
        plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {scatter_plot_path}")
        plt.close(fig_sc) # Close the figure
    else:
         logging.warning("Skipping scatter plot due to insufficient data.")
         scatter_plot_path = None

    # Return dict of plot paths
    plot_files = {}
    if ts_plot_path: plot_files['time_series'] = ts_plot_path
    if scatter_plot_path: plot_files['scatter'] = scatter_plot_path
    return plot_files

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