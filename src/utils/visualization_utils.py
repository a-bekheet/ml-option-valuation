# src/utils/visualization_utils.py

import logging
import os
import tqdm
from typing import Optional, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path # Use pathlib
from datetime import datetime
import torch # Keep torch import if needed elsewhere, though not strictly for plotting

def save_and_display_results(model, history, analysis, ticker, target_cols,
                             models_dir="models", # This can now come from config['models_dir']
                             plots_dir="plots"   # Separate dir for plots, from config['viz_dir']
                             ):
    """
    Save the model and training plots to specified directories.
    Uses pathlib for robust path handling.
    """
    models_path = Path(models_dir)
    plots_path = Path(plots_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Changed timestamp format slightly
    target_str = "-".join(target_cols)

    # Save model
    # Model filename includes architecture details inferred from class name if possible
    arch_name = model.__class__.__name__.replace('Model', '') # e.g., HybridRNN, GRUGRU
    model_save_path = models_path / f"{arch_name}_{ticker}_target_{target_str}_{timestamp}.pth"
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"\nModel saved to {model_save_path}")
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"\nError saving model: {e}")
        logging.error(f"Failed to save model to {model_save_path}: {e}")


    # Plot training history
    try:
        plt.figure(figsize=(10, 6))
        if 'train_losses' in history and history['train_losses']:
            plt.plot(history['train_losses'], label="Train Loss", linewidth=2)
        if 'val_losses' in history and history['val_losses']:
            plt.plot(history['val_losses'], label="Validation Loss", linewidth=2)
        plt.title(f"Training & Validation Loss ({ticker} - {target_str})")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True, alpha=0.5)

        plot_save_path = plots_path / f"training_plot_{ticker}_{target_str}_{timestamp}.png"
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {plot_save_path}")
        logging.info(f"Training plot saved to {plot_save_path}")
        plt.close() # Close figure to free memory
    except Exception as e:
         print(f"\nError saving training plot: {e}")
         logging.error(f"Failed to save training plot: {e}")

    # Plot predicted vs actual values if available in history (usually from test set)
    if 'y_true' in history and 'y_pred' in history and history['y_true'] is not None and history['y_pred'] is not None:
         try:
             print("\nGenerating Test Set Prediction Plot...")
             # Assuming y_true/y_pred are numpy arrays or lists
             y_true_np = np.array(history['y_true'])
             y_pred_np = np.array(history['y_pred'])

             # Pass metrics if available in history['test_metrics']
             rmse = history.get('test_metrics', {}).get('rmse')
             mae = history.get('test_metrics', {}).get('mae')

             # Call plot_predictions, ensuring output_dir is plots_path
             plot_predictions(
                 y_true=y_true_np,
                 y_pred=y_pred_np,
                 target_cols=target_cols,
                 ticker=ticker,
                 output_dir=str(plots_path), # Pass path as string
                 timestamp=timestamp, # Reuse timestamp
                 rmse=rmse,
                 mae=mae,
                 n_samples_plot=250 # Or get from config if needed
             )
         except Exception as e:
             print(f"\nError generating prediction plot from history: {e}")
             logging.error(f"Failed to generate prediction plot from history: {e}")


    # Display Model Analysis (remains the same)
    display_model_analysis(analysis)

def display_model_analysis(analysis):
    """Display the model architecture analysis in a formatted way."""
    # Check if analysis is valid
    if not analysis or not isinstance(analysis, dict):
         print("\nModel Architecture Analysis: Not Available")
         return

    print("\nNetwork Architecture Analysis:")
    print("-" * 50)
    total_params = analysis.get('total_parameters', 'N/A')
    trainable_params = analysis.get('trainable_parameters', 'N/A')

    # Format parameter counts with commas if they are integers
    total_params_str = f"{total_params:,}" if isinstance(total_params, int) else str(total_params)
    trainable_params_str = f"{trainable_params:,}" if isinstance(trainable_params, int) else str(trainable_params)

    print(f"Total parameters: {total_params_str}")
    print(f"Trainable parameters: {trainable_params_str}")

    layer_shapes = analysis.get('layer_shapes')
    if isinstance(layer_shapes, dict) and layer_shapes: # Check if it's a non-empty dict
        print("\nLayer Shapes (Input -> Output):")
        for layer_name, shapes in layer_shapes.items():
            # Check if shapes is a dictionary before accessing keys
            if isinstance(shapes, dict):
                in_shape = shapes.get('input_shape', 'N/A')
                out_shape = shapes.get('output_shape', 'N/A')
                print(f"  {layer_name}:")
                print(f"    Input: {in_shape}")
                print(f"    Output: {out_shape}")
            else:
                 # Handle cases where shapes might be an error string or None
                 print(f"  {layer_name}: Shape details not available ({shapes})")
    elif layer_shapes: # Handle case where it might be an error string etc.
         print(f"\nLayer Shapes: {layer_shapes}")
    else:
         print("\nLayer Shapes: Details not available.")
    print("-" * 50)


def plot_predictions(y_true, y_pred, target_cols, ticker,
                     output_dir: str, # Now explicitly required
                     timestamp: Optional[str] = None,
                     rmse: Optional[float] = None,
                     mae: Optional[float] = None,
                     n_samples_plot: int = 250,
                     plot_suffix: str = "" # Optional suffix for filenames
                     ) -> Dict[str, str]:
    """
    Plot predicted vs actual values for each target column using original scale,
    annotate with error metrics, and save to specified output directory using pathlib.

    Args:
        y_true: Numpy array of true values (original scale recommended).
        y_pred: Numpy array of predicted values (original scale recommended).
        target_cols: List of target column names.
        ticker: Stock ticker symbol.
        output_dir (str): Directory path (as string) to save plots.
        timestamp (Optional[str]): Timestamp for filenames. Defaults to current time.
        rmse (Optional[float]): Root Mean Squared Error (original scale) for annotation.
        mae (Optional[float]): Mean Absolute Error (original scale) for annotation.
        n_samples_plot (int): Max number of samples for time series plot.
        plot_suffix (str): Optional suffix to add before the timestamp in filenames.

    Returns:
        Dict[str, str]: Dictionary mapping plot types to saved absolute file paths.
    """
    output_path = Path(output_dir) # Convert string path to Path object
    output_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure inputs are numpy arrays
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Ensure proper shapes (samples x targets)
    if y_true_np.ndim == 1: y_true_np = y_true_np.reshape(-1, 1)
    if y_pred_np.ndim == 1: y_pred_np = y_pred_np.reshape(-1, 1)

    num_targets = y_true_np.shape[1]
    if len(target_cols) != num_targets:
         logging.warning(f"Mismatch between target_cols ({len(target_cols)}) and data shape ({num_targets}). Using generic target names.")
         target_cols = [f"Target_{i+1}" for i in range(num_targets)]

    plot_files = {}

    # --- Time Series Plot ---
    num_samples = min(n_samples_plot, len(y_true_np))
    indices = np.linspace(0, len(y_true_np)-1, num_samples, dtype=int) if len(y_true_np) > 0 else []

    if len(indices) > 0: # Only plot if there's data
        try:
            fig_ts, axs_ts = plt.subplots(num_targets, 1, figsize=(12, 5 * num_targets), dpi=100, squeeze=False) # Ensure axs is 2D

            for i in range(num_targets):
                target_name = target_cols[i]
                axs_ts[i, 0].plot(indices, y_true_np[indices, i], 'b-', label=f'Actual {target_name}', linewidth=1.5)
                axs_ts[i, 0].plot(indices, y_pred_np[indices, i], 'r--', label=f'Predicted {target_name}', linewidth=1.5)
                axs_ts[i, 0].set_title(f'{ticker} - {target_name} Prediction vs Actual (Sample)')
                axs_ts[i, 0].set_xlabel('Sample Index')
                axs_ts[i, 0].set_ylabel(f'{target_name} Value') # Removed ($) - might be normalized
                axs_ts[i, 0].legend()
                axs_ts[i, 0].grid(True, alpha=0.5)

            plt.tight_layout()
            ts_filename = f"{ticker}_preds_ts{plot_suffix}_{timestamp}.png"
            ts_plot_path = output_path / ts_filename
            plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
            plot_files['time_series'] = str(ts_plot_path.resolve()) # Store absolute path
            logging.info(f"Time series plot saved to {ts_plot_path}")
            plt.close(fig_ts) # Close the figure
        except Exception as e:
            logging.error(f"Failed to generate time series plot: {e}")
            plt.close(fig_ts) # Ensure figure is closed on error
    else:
         logging.warning("Skipping time series plot due to insufficient data.")


    # --- Scatter Plot ---
    if len(y_true_np) > 0: # Only plot if there's data
        try:
            fig_sc, axs_sc = plt.subplots(1, num_targets, figsize=(7 * num_targets, 6), dpi=100, squeeze=False) # Ensure axs is 2D

            for i in range(num_targets):
                target_name = target_cols[i]
                axs_sc[0, i].scatter(y_true_np[:, i], y_pred_np[:, i], alpha=0.3, s=15, edgecolors='k', linewidths=0.5) # Improved aesthetics

                # Add perfect prediction line (y=x)
                min_val = min(np.nanmin(y_true_np[:, i]), np.nanmin(y_pred_np[:, i]))
                max_val = max(np.nanmax(y_true_np[:, i]), np.nanmax(y_pred_np[:, i]))
                # Add buffer to min/max for line limits
                buffer = (max_val - min_val) * 0.05 if max_val > min_val else 1
                line_lims = [min_val - buffer, max_val + buffer]
                axs_sc[0, i].plot(line_lims, line_lims, 'k--', label='y=x (Perfect)', linewidth=1)

                axs_sc[0, i].set_title(f'{ticker} - {target_name}')
                axs_sc[0, i].set_xlabel(f'Actual {target_name}') # Removed ($)
                axs_sc[0, i].set_ylabel(f'Predicted {target_name}') # Removed ($)
                axs_sc[0, i].grid(True, alpha=0.5)
                axs_sc[0, i].set_xlim(line_lims) # Set limits based on data range
                axs_sc[0, i].set_ylim(line_lims)

                # Add Metric Annotations
                annotation_text = []
                if rmse is not None and not np.isnan(rmse): annotation_text.append(f'RMSE: {rmse:.4f}')
                if mae is not None and not np.isnan(mae): annotation_text.append(f'MAE:  {mae:.4f}')

                if annotation_text:
                     # Place text using axes coordinates
                     axs_sc[0, i].text(0.05, 0.95, "\n".join(annotation_text),
                                       transform=axs_sc[0, i].transAxes,
                                       fontsize=9, verticalalignment='top',
                                       bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

                axs_sc[0, i].legend(loc='lower right') # Adjust legend position

            plt.tight_layout()
            scatter_filename = f"{ticker}_scatter{plot_suffix}_{timestamp}.png"
            scatter_plot_path = output_path / scatter_filename
            plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
            plot_files['scatter'] = str(scatter_plot_path.resolve()) # Store absolute path
            logging.info(f"Scatter plot saved to {scatter_plot_path}")
            plt.close(fig_sc) # Close the figure
        except Exception as e:
            logging.error(f"Failed to generate scatter plot: {e}")
            plt.close(fig_sc) # Ensure figure is closed on error
    else:
         logging.warning("Skipping scatter plot due to insufficient data.")

    return plot_files

# --- plot_model_predictions function remains the same conceptually ---
# ... (Paste your existing plot_model_predictions function here, ensure it uses the updated plot_predictions) ...
def plot_model_predictions(model, data_loader, target_cols, ticker,
                           output_dir="plots", # Use separate plots dir
                           device='cpu'):
    """
    Run model predictions on a data loader and plot predicted vs actual values.
    Assumes data from loader is normalized, plots will show normalized scale.

    Args:
        model: PyTorch model to use for predictions
        data_loader: DataLoader with test data
        target_cols: List of target column names
        ticker: Stock ticker symbol
        output_dir: Directory path (string) to save plots
        device: Device to run model on ('cpu', 'cuda', 'mps')

    Returns:
        Tuple of numpy arrays (y_true_norm, y_pred_norm)
    """
    model.eval()
    all_y_true_norm = []
    all_y_pred_norm = []
    model.to(device) # Ensure model is on the correct device

    print(f"\nGenerating predictions for plot using device: {device}")
    with torch.no_grad():
        for x_seq, y_val in tqdm(data_loader, desc="Plot Predictions", leave=False):
            x_seq = x_seq.to(device) # Move input sequence to device
            # y_val (true value) can stay on CPU
            y_pred = model(x_seq) # Prediction happens on device
            all_y_true_norm.extend(y_val.cpu().numpy()) # Ensure true values are on CPU for numpy conversion
            all_y_pred_norm.extend(y_pred.cpu().numpy()) # Move predictions to CPU for numpy conversion

    y_true_norm_np = np.array(all_y_true_norm)
    y_pred_norm_np = np.array(all_y_pred_norm)

    # Call the updated plotting function
    plot_predictions(
        y_true=y_true_norm_np,
        y_pred=y_pred_norm_np,
        target_cols=target_cols,
        ticker=ticker,
        output_dir=output_dir, # Pass the directory path
        # Note: RMSE/MAE are not calculated here, so annotations won't appear unless calculated separately
        plot_suffix="_direct" # Add suffix to distinguish these plots
    )

    return y_true_norm_np, y_pred_norm_np