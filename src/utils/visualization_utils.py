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
import seaborn as sns # Import Seaborn

def save_and_display_results(model, history, analysis, ticker, target_cols,
                             models_dir="models", plots_dir="plots"):
    """Saves model, plots training history, and optionally prediction plots."""
    # Ensure torch is imported if used
    import torch
    models_path = Path(models_dir)
    plots_path = Path(plots_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_str = "-".join(target_cols)
    arch_name = model.__class__.__name__.replace('Model', '')

    # Save model
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
        # Apply a style for better thesis visuals
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))
        if 'train_losses' in history and history['train_losses']:
            plt.plot(history['train_losses'], label="Training Loss", linewidth=2, color='royalblue')
        if 'val_losses' in history and history['val_losses']:
            plt.plot(history['val_losses'], label="Validation Loss", linewidth=2, color='darkorange')

        plt.title(f"Model Training and Validation Loss ({ticker} - {target_str})", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Mean Squared Error (MSE) Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        plot_save_path = plots_path / f"training_plot_{ticker}_{target_str}_{timestamp}.png"
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {plot_save_path}")
        logging.info(f"Training plot saved to {plot_save_path}")
        plt.close()
    except Exception as e:
         print(f"\nError saving training plot: {e}")
         logging.error(f"Failed to save training plot: {e}")
         plt.close() # Ensure closure on error

    # Plot test set predictions if available
    if 'y_true' in history and 'y_pred' in history and history['y_true'] is not None and history['y_pred'] is not None:
         try:
             print("\nGenerating Test Set Prediction Plots...")
             y_true_np = np.array(history['y_true'])
             y_pred_np = np.array(history['y_pred'])
             rmse = history.get('test_metrics', {}).get('rmse')
             mae = history.get('test_metrics', {}).get('mae')

             # Call updated plot_predictions
             plot_predictions(
                 y_true=y_true_np, y_pred=y_pred_np, target_cols=target_cols,
                 ticker=ticker, output_dir=str(plots_path), timestamp=timestamp,
                 rmse=rmse, mae=mae, n_samples_plot=250, plot_suffix="_test_set"
             )
         except Exception as e:
             print(f"\nError generating prediction plot from history: {e}")
             logging.error(f"Failed to generate prediction plot from history: {e}")

    # Display Model Analysis (remains the same)
    display_model_analysis(analysis)

def display_model_analysis(analysis):
    """Display the model architecture analysis in a formatted way."""
    # ... (implementation remains the same as provided previously) ...
    if not analysis or not isinstance(analysis, dict): print("\nModel Architecture Analysis: Not Available"); return
    print("\nNetwork Architecture Analysis:"); print("-" * 50)
    total_params=analysis.get('total_parameters','N/A'); trainable_params=analysis.get('trainable_parameters','N/A')
    total_params_str=f"{total_params:,}" if isinstance(total_params,int) else str(total_params)
    trainable_params_str=f"{trainable_params:,}" if isinstance(trainable_params,int) else str(trainable_params)
    print(f"Total parameters: {total_params_str}"); print(f"Trainable parameters: {trainable_params_str}")
    layer_shapes=analysis.get('layer_shapes')
    if isinstance(layer_shapes,dict) and layer_shapes:
        print("\nLayer Shapes (Input -> Output):")
        for layer_name, shapes in layer_shapes.items():
            if isinstance(shapes,dict):
                in_shape=shapes.get('input_shape','N/A'); out_shape=shapes.get('output_shape','N/A')
                print(f"  {layer_name}:"); print(f"    Input: {in_shape}"); print(f"    Output: {out_shape}")
            else: print(f"  {layer_name}: Shape details not available ({shapes})")
    elif layer_shapes: print(f"\nLayer Shapes: {layer_shapes}")
    else: print("\nLayer Shapes: Details not available.")
    print("-" * 50)



def plot_predictions(y_true, y_pred, target_cols, ticker,
                     output_dir: str,
                     timestamp: Optional[str] = None,
                     rmse: Optional[float] = None,
                     mae: Optional[float] = None,
                     n_samples_plot: int = 250,
                     plot_suffix: str = ""
                     ) -> Dict[str, str]:
    """
    Generate enhanced prediction plots (Time Series and 2D Histogram) suitable for thesis.

    Args:
        y_true (np.ndarray): True values (original scale recommended).
        y_pred (np.ndarray): Predicted values (original scale recommended).
        target_cols (List[str]): Target column names.
        ticker (str): Stock ticker symbol.
        output_dir (str): Directory path (string) to save plots.
        timestamp (Optional[str]): Timestamp for filenames.
        rmse (Optional[float]): RMSE metric for annotation.
        mae (Optional[float]): MAE metric for annotation.
        n_samples_plot (int): Number of samples for the time series plot.
        plot_suffix (str): Optional suffix for filenames (e.g., "_test_set").

    Returns:
        Dict[str, str]: Dictionary mapping plot types to saved absolute file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Apply a professional style
    plt.style.use('seaborn-v0_8-whitegrid')

    y_true_np = np.array(y_true); y_pred_np = np.array(y_pred)
    if y_true_np.ndim == 1: y_true_np = y_true_np.reshape(-1, 1)
    if y_pred_np.ndim == 1: y_pred_np = y_pred_np.reshape(-1, 1)

    num_targets = y_true_np.shape[1]
    if len(target_cols) != num_targets:
         logging.warning(f"Target cols/data shape mismatch. Using generic names.")
         target_cols = [f"Target_{i+1}" for i in range(num_targets)]

    plot_files = {}
    value_label = "Option Price" # Generic label, assumes price prediction

    # --- Time Series Plot ---
    num_samples = min(n_samples_plot, len(y_true_np))
    indices = np.linspace(0, len(y_true_np)-1, num_samples, dtype=int) if len(y_true_np) > 0 else []

    if len(indices) > 0:
        try:
            fig_ts, axs_ts = plt.subplots(num_targets, 1, figsize=(12, 4 * num_targets + 1), dpi=120, squeeze=False)
            fig_ts.suptitle(f'Model Predictions vs. Actual Values Over Time (Sample) - {ticker}', fontsize=16, y=1.02)

            for i in range(num_targets):
                target_name = target_cols[i]
                ax = axs_ts[i, 0]
                ax.plot(indices, y_true_np[indices, i], 'o-', label=f'Actual {target_name}', color='royalblue', markersize=4, linewidth=1.5)
                ax.plot(indices, y_pred_np[indices, i], 'x--', label=f'Predicted {target_name}', color='darkorange', markersize=5, linewidth=1.5)
                ax.set_title(f'{target_name.capitalize()} Predictions', fontsize=14)
                ax.set_xlabel('Sample Index (Chronological Subset)', fontsize=12)
                ax.set_ylabel(value_label, fontsize=12)
                ax.legend(fontsize=11, loc='best')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.tick_params(axis='both', which='major', labelsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout for suptitle
            ts_filename = f"{ticker}_timeseries{plot_suffix}_{timestamp}.png"
            ts_plot_path = output_path / ts_filename
            plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
            plot_files['time_series'] = str(ts_plot_path.resolve())
            logging.info(f"Time series plot saved to {ts_plot_path}")
            plt.close(fig_ts)
        except Exception as e:
            logging.error(f"Failed to generate time series plot: {e}"); plt.close('all')
    else: logging.warning("Skipping time series plot (insufficient data).")


    # --- 2D Histogram / Heatmap ---
    if len(y_true_np) > 0:
        try:
            fig_hist, axs_hist = plt.subplots(1, num_targets, figsize=(7 * num_targets, 6.5), dpi=120, squeeze=False)
            fig_hist.suptitle(f'Prediction Density: Actual vs. Predicted Values - {ticker}', fontsize=16, y=1.03)

            for i in range(num_targets):
                target_name = target_cols[i]
                ax = axs_hist[0, i]

                # Create the 2D Histogram using Seaborn
                # Adjust bins as needed, cmap options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', etc.
                hb = sns.histplot(ax=ax, x=y_true_np[:, i], y=y_pred_np[:, i], bins=50, cmap="Blues", cbar=True, cbar_kws={'label': 'Density (Count)'})
                # Alternatively, use plt.hist2d:
                # counts, xedges, yedges, im = ax.hist2d(y_true_np[:, i], y_pred_np[:, i], bins=50, cmap='viridis', cmin=1) # cmin avoids empty bins
                # fig_hist.colorbar(im, ax=ax, label='Density (Count)')

                # Add y=x line for reference
                min_val = min(np.nanmin(y_true_np[:, i]), np.nanmin(y_pred_np[:, i]))
                max_val = max(np.nanmax(y_true_np[:, i]), np.nanmax(y_pred_np[:, i]))
                buffer = (max_val - min_val) * 0.05 if max_val > min_val else 1
                line_lims = [min_val - buffer, max_val + buffer]
                ax.plot(line_lims, line_lims, 'r--', label='y = x (Perfect Prediction)', linewidth=1.5)

                ax.set_title(f'{target_name.capitalize()} Density', fontsize=14)
                ax.set_xlabel(f'Actual {value_label}', fontsize=12)
                ax.set_ylabel(f'Predicted {value_label}', fontsize=12)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlim(line_lims); ax.set_ylim(line_lims) # Ensure equal aspect ratio might be needed depending on data range
                ax.tick_params(axis='both', which='major', labelsize=10)

                # Add Metric Annotations
                annotation_text = []
                if rmse is not None and not np.isnan(rmse): annotation_text.append(f'RMSE: {rmse:.4f}')
                if mae is not None and not np.isnan(mae): annotation_text.append(f'MAE:  {mae:.4f}')
                if annotation_text:
                     ax.text(0.05, 0.95, "\n".join(annotation_text), transform=ax.transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.8))

                ax.legend(loc='lower right', fontsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            hist_filename = f"{ticker}_density_heatmap{plot_suffix}_{timestamp}.png"
            hist_plot_path = output_path / hist_filename
            plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
            plot_files['density_heatmap'] = str(hist_plot_path.resolve())
            logging.info(f"Density heatmap saved to {hist_plot_path}")
            plt.close(fig_hist)
        except Exception as e:
            logging.error(f"Failed to generate density heatmap: {e}"); plt.close('all')
    else: logging.warning("Skipping density heatmap (insufficient data).")

    # Reset style to default if necessary, or manage styles via context manager
    # plt.style.use('default')

    return plot_files

def plot_model_predictions(model, data_loader, target_cols, ticker,
                           output_dir="plots", device='cpu'):
    # ... (Implementation remains the same as provided previously) ...
    model.eval(); all_y_true_norm=[]; all_y_pred_norm=[]
    model.to(device)
    print(f"\nGenerating predictions for plot using device: {device}")
    with torch.no_grad():
        for x_seq, y_val in tqdm(data_loader, desc="Plot Predictions", leave=False):
            x_seq=x_seq.to(device); y_pred=model(x_seq)
            all_y_true_norm.extend(y_val.cpu().numpy()); all_y_pred_norm.extend(y_pred.cpu().numpy())
    y_true_norm_np = np.array(all_y_true_norm); y_pred_norm_np = np.array(all_y_pred_norm)
    # Call the updated plotting function
    plot_predictions(y_true=y_true_norm_np, y_pred=y_pred_norm_np, target_cols=target_cols,
                     ticker=ticker, output_dir=output_dir, plot_suffix="_direct")
    return y_true_norm_np, y_pred_norm_np