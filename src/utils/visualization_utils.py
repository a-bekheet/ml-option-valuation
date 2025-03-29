# src/utils/visualization_utils.py

import logging
import os
from scipy import stats
import tqdm
from typing import Optional, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path # Use pathlib
from datetime import datetime
import torch # Keep torch import if needed elsewhere, though not strictly for plotting
import seaborn as sns # Import Seaborn
from .model_utils import load_scaling_params, recover_original_values

def save_and_display_results(model, history, analysis, ticker, target_cols,
                             models_dir="models", plots_dir="plots",
                             data_dir=None): # <-- data_dir parameter added
    """Saves model, plots training history, and optionally prediction plots."""
    # Ensure torch is imported if used
    import torch
    models_path = Path(models_dir)
    plots_path = Path(plots_dir) # <-- plots_path defined here
    models_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # <-- timestamp defined here
    target_str = "-".join(target_cols)
    arch_name = model.__class__.__name__.replace('Model', '')

    # --- Save Model ---
    model_save_path = models_path / f"{arch_name}_{ticker}_target_{target_str}_{timestamp}.pth"
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"\nModel saved to {model_save_path}")
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"\nError saving model: {e}")
        logging.error(f"Failed to save model to {model_save_path}: {e}")

    # --- Plot Training History ---
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

    # --- Plot test set predictions (potentially un-normalized) ---
    if ('y_true' in history and 'y_pred' in history and
        history['y_true'] is not None and history['y_pred'] is not None and
        len(history['y_true']) > 0 and len(history['y_pred']) > 0):
        try:
            print("\nProcessing Test Set Predictions for Plotting...")
            y_true_norm = np.array(history['y_true'])
            y_pred_norm = np.array(history['y_pred'])

            # Ensure arrays are 2D
            if y_true_norm.ndim == 1: y_true_norm = y_true_norm.reshape(-1, 1)
            if y_pred_norm.ndim == 1: y_pred_norm = y_pred_norm.reshape(-1, 1)

            # --- Add Un-normalization Logic ---
            y_true_plot = y_true_norm # Default to normalized if recovery fails
            y_pred_plot = y_pred_norm
            scaling_params = None
            un_normalized = False

            print(f"DEBUG (in save_and_display_results): Checking condition data_dir='{data_dir}', ticker='{ticker}'")

            if data_dir and ticker:
                # Use the imported function
                scaling_params = load_scaling_params(ticker, data_dir)
                if scaling_params:
                    try:
                        print(f"Attempting to recover original scale for {ticker} using params from {data_dir}")
                        # Use the target_cols provided to the function
                        # Use the imported function
                        y_true_plot = recover_original_values(y_true_norm, target_cols, scaling_params)
                        y_pred_plot = recover_original_values(y_pred_norm, target_cols, scaling_params)
                        print("Successfully recovered original scale values.")
                        un_normalized = True
                    except Exception as recovery_err:
                        print(f"\nWarning: Failed to recover original values: {recovery_err}")
                        logging.warning(f"Failed recovery for {ticker}: {recovery_err}")
                else:
                    print("\nWarning: Scaling parameters not found. Plotting normalized values.")
            else:
                print("\nWarning: data_dir or ticker not provided for un-normalization. Plotting normalized values.")
            # --- End Un-normalization Logic ---

            # Get metrics (note: these metrics might be based on normalized scale if calculated before un-normalization)
            rmse = history.get('test_metrics', {}).get('rmse')
            mae = history.get('test_metrics', {}).get('mae')
            metrics_scale_note = "(Original Scale)" if un_normalized else "(Normalized Scale)"
            print(f"Using metrics {metrics_scale_note} for plot annotations: RMSE={rmse}, MAE={mae}")

            # Call plot_predictions with the data (original or normalized scale)
            print(f"Generating plots using {'original' if un_normalized else 'normalized'} scale data...")
            plot_predictions(
                y_true=y_true_plot,
                y_pred=y_pred_plot,
                target_cols=target_cols,
                ticker=ticker,
                output_dir=str(plots_path), # Use defined plots_path
                timestamp=timestamp,       # Use defined timestamp
                rmse=rmse,
                mae=mae,
                n_samples_plot=250,
                plot_suffix="_test_set"
            )
        except Exception as e:
            print(f"\nError generating prediction plot from history: {e}")
            logging.error(f"Failed to generate prediction plot from history: {e}", exc_info=True)
    elif 'y_true' in history and 'y_pred' in history:
         print("\nSkipping Test Set Prediction Plots: No test data found in history.")

    # --- Display Model Analysis ---
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
                     n_samples_plot: int = 250, # Note: Not currently used for sampling in this version
                     plot_suffix: str = ""
                     ) -> Dict[str, str]:
    """
    Generate thesis-quality prediction plots using only matplotlib with full data:
    1. Time series comparison showing prediction accuracy over time (all data points)
    2. 2D histogram showing density of prediction vs. actual values
    3. Enhanced error distribution plots with professional styling, dynamic x-axis,
       and consistent annotation placement.

    Args:
        y_true (np.ndarray): True values (original scale recommended).
        y_pred (np.ndarray): Predicted values (original scale recommended).
        target_cols (List[str]): Target column names.
        ticker (str): Stock ticker symbol.
        output_dir (str): Directory path to save plots.
        timestamp (Optional[str]): Timestamp for filenames. If None, generated automatically.
        rmse (Optional[float]): RMSE metric for annotation.
        mae (Optional[float]): MAE metric for annotation.
        n_samples_plot (int): Number of samples for time series plot (if sampling is reintroduced).
                              Currently plots all data points.
        plot_suffix (str): Optional suffix for filenames (e.g., "_test_set").

    Returns:
        Dict[str, str]: Dictionary mapping plot types to saved file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Consistent style for annotation boxes
    bbox_style = dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8, edgecolor='darkgray')

    # Set default matplotlib style - clean and professional
    # Using a context manager to avoid affecting global settings if not desired
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'font.size': 11,
            'legend.frameon': True, # Ensure legend has a frame
            'legend.framealpha': 0.8,
            'legend.facecolor': 'white',
            'legend.edgecolor': 'darkgray' # Match annotation box edge
        })

        # Ensure arrays are properly formatted
        try:
            y_true_np = np.array(y_true)
            y_pred_np = np.array(y_pred)

            if y_true_np.ndim == 1:
                y_true_np = y_true_np.reshape(-1, 1)
            if y_pred_np.ndim == 1:
                y_pred_np = y_pred_np.reshape(-1, 1)

            # Validate dimensions match
            if y_true_np.shape[1] != y_pred_np.shape[1]:
                logging.error(f"Dimension mismatch: y_true has {y_true_np.shape[1]} targets, y_pred has {y_pred_np.shape[1]} targets")
                return {}
            num_targets = y_true_np.shape[1]

            # Validate target columns
            if len(target_cols) != num_targets:
                logging.warning(f"Target columns ({len(target_cols)}) don't match data dimensions ({num_targets}). Using generic names.")
                target_cols = [f"Target_{i+1}" for i in range(num_targets)]

        except Exception as e:
             logging.error(f"Error processing input arrays: {e}")
             return {}


        plot_files = {}
        data_size = len(y_true_np)
        logging.info(f"Plotting ticker {ticker} with {data_size} data points")

        # --- TIME SERIES PLOT ---
        try:
            indices = np.arange(data_size) # Use all data points

            # Adjust line/marker based on data size
            marker_size = 0
            if data_size <= 500: line_width = 1.5
            elif data_size <= 5000: line_width = 1.2
            else: line_width = 1.0

            fig_ts, axs_ts = plt.subplots(num_targets, 1, figsize=(12, 4 * num_targets), dpi=300, squeeze=False)
            fig_ts.suptitle(f'Temporal Prediction Performance for {ticker} Options', fontsize=16)

            for i in range(num_targets):
                target_name = target_cols[i]
                ax = axs_ts[i, 0]

                ax.plot(indices, y_true_np[:, i], '-', label=f'Actual {target_name}',
                       color='#1f77b4', linewidth=line_width)
                ax.plot(indices, y_pred_np[:, i], '--', label=f'Predicted {target_name}',
                       color='#ff7f0e', linewidth=line_width)

                if data_size <= 10000: # Add error shading only if not too dense
                    ax.fill_between(indices, y_true_np[:, i], y_pred_np[:, i],
                                   color='#ff7f0e', alpha=0.15, interpolate=True)

                ax.set_title(f'Sequential {target_name.title()} Prediction Analysis', fontsize=14)
                ax.set_xlabel('Sample Index (Chronological Sequence)', fontsize=12)
                ax.set_ylabel(f'{target_name.title()} Value ($)', fontsize=12)
                ax.legend(fontsize=11, loc='best') # Keep legend location flexible here
                ax.grid(True, linestyle='--', alpha=0.7)

                # Add metrics annotation if available (using consistent style)
                if rmse is not None or mae is not None:
                    metric_text = []
                    if rmse is not None and np.isfinite(rmse): metric_text.append(f'RMSE: {rmse:.4f}')
                    if mae is not None and np.isfinite(mae): metric_text.append(f'MAE: {mae:.4f}')

                    if metric_text:
                        ax.annotate('\n'.join(metric_text), xy=(0.02, 0.96), xycoords='axes fraction',
                                  bbox=bbox_style, fontsize=10, ha='left', va='top')

            plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for suptitle
            ts_filename = f"{ticker}_timeseries{plot_suffix}_{timestamp}.png"
            ts_plot_path = output_path / ts_filename
            plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
            plot_files['time_series'] = str(ts_plot_path.resolve())
            logging.info(f"Time series plot saved to {ts_plot_path}")
            plt.close(fig_ts)
        except Exception as e:
            logging.error(f"Failed to generate time series plot: {e}", exc_info=True)
            plt.close('all')

        # --- 2D HISTOGRAM PREDICTION ANALYSIS ---
        try:
            for i in range(num_targets):
                target_name = target_cols[i]
                true_values = y_true_np[:, i]
                pred_values = y_pred_np[:, i]

                # Remove NaNs for histogram calculation
                valid_mask_hist = ~np.isnan(true_values) & ~np.isnan(pred_values)
                true_values_clean = true_values[valid_mask_hist]
                pred_values_clean = pred_values[valid_mask_hist]

                if len(true_values_clean) == 0:
                    logging.warning(f"Skipping 2D histogram for {target_name}: No valid data points.")
                    continue

                fig_hist = plt.figure(figsize=(8, 7), dpi=300)
                ax_hist = fig_hist.add_subplot(111)
                ax_hist.set_title(f'Prediction Accuracy Distribution for {ticker} {target_name.title()}', fontsize=14)

                # Determine bins
                hist_data_size = len(true_values_clean)
                if hist_data_size < 500: bins = 25
                elif hist_data_size < 5000: bins = 50
                else: bins = 75 # Adjusted default

                # Determine range
                min_val = min(np.nanmin(true_values_clean), np.nanmin(pred_values_clean))
                max_val = max(np.nanmax(true_values_clean), np.nanmax(pred_values_clean))
                data_range = max_val - min_val
                buffer = data_range * 0.05 if data_range > 0 else 1.0 # Add buffer, handle zero range
                plot_range = [[min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer]]

                # Create 2D histogram
                h = ax_hist.hist2d(true_values_clean, pred_values_clean,
                                bins=bins,
                                range=plot_range,
                                cmap='viridis',
                                norm=plt.cm.colors.LogNorm(), # Log scale for better visibility
                                cmin=1) # Minimum count to display a bin

                fig_hist.colorbar(h[3], ax=ax_hist, label='Point Density (log scale)')

                # Perfect prediction line
                ax_hist.plot(plot_range[0], plot_range[0], 'k--', linewidth=1.5, label='Perfect Prediction')

                # Regression line
                try:
                    if len(true_values_clean) > 2: # Need >2 points for polyfit
                        z = np.polyfit(true_values_clean, pred_values_clean, 1)
                        p = np.poly1d(z)
                        ax_hist.plot(plot_range[0], p(plot_range[0]), 'r-', linewidth=1.5,
                                label=f'Regression (y={z[0]:.2f}x+{z[1]:.2f})')
                except Exception as fit_err:
                    logging.warning(f"Could not calculate regression line for {target_name}: {fit_err}")


                # --- POSITIONING & STYLING ---
                # Legend -> Top-Left
                ax_hist.legend(loc='upper left', fontsize=10) # Uses default style set by rcParams

                ax_hist.set_xlabel(f'Actual {target_name} ($)', fontsize=12)
                ax_hist.set_ylabel(f'Predicted {target_name} ($)', fontsize=12)

                # Metrics annotation -> Top-Right
                if rmse is not None or mae is not None:
                    metric_text = []
                    if rmse is not None and np.isfinite(rmse): metric_text.append(f'RMSE: {rmse:.4f}')
                    if mae is not None and np.isfinite(mae): metric_text.append(f'MAE: {mae:.4f}')

                    # Calculate R²
                    if len(true_values_clean) > 1:
                        try:
                            # Ensure shapes are compatible if needed, though corrcoef handles 1D arrays
                            r_matrix = np.corrcoef(true_values_clean, pred_values_clean)
                            # Check if correlation matrix is valid
                            if r_matrix.shape == (2, 2) and np.all(np.isfinite(r_matrix)):
                                r2 = r_matrix[0, 1]**2
                                metric_text.append(f'$R^2$: {r2:.4f}') # Use LaTeX for R-squared
                            else:
                                logging.warning(f"Could not calculate R^2 for {target_name}: Invalid correlation matrix.")
                        except Exception as r2_err:
                            logging.warning(f"Could not calculate R^2 for {target_name}: {r2_err}")

                    if metric_text:
                        ax_hist.annotate('\n'.join(metric_text),
                                    xy=(0.97, 0.97), # Positioned top-right
                                    xycoords='axes fraction',
                                    bbox=bbox_style, # Use consistent style
                                    fontsize=9,
                                    ha='right',
                                    va='top')

                ax_hist.set_aspect('equal') # Keep aspect ratio equal
                ax_hist.grid(True, linestyle='--', alpha=0.5)

                hist_filename = f"{ticker}_{target_name}_2dhist{plot_suffix}_{timestamp}.png"
                hist_plot_path = output_path / hist_filename
                plt.tight_layout()
                plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
                plot_files[f'hist2d_{target_name}'] = str(hist_plot_path.resolve())
                logging.info(f"2D histogram for {target_name} saved to {hist_plot_path}")
                plt.close(fig_hist)

        except Exception as e:
            logging.error(f"Failed to generate 2D histogram plot: {e}", exc_info=True)
            plt.close('all')

        # --- ERROR DISTRIBUTION PLOTS ---
        try:
            for i in range(num_targets):
                target_name = target_cols[i]
                true_values = y_true_np[:, i]
                pred_values = y_pred_np[:, i]

                # Calculate errors (now on original scale)
                errors = pred_values - true_values
                errors_clean = errors[~np.isnan(errors)] # Remove NaNs for calculations

                if len(errors_clean) == 0:
                    logging.warning(f"Skipping error distribution plot for {target_name}: No valid error points.")
                    continue

                fig_err = plt.figure(figsize=(10, 6), dpi=300)
                ax_err = fig_err.add_subplot(111)

                # Calculate error statistics
                mean_error = np.mean(errors_clean)
                median_error = np.median(errors_clean)
                std_error = np.std(errors_clean)

                # Determine bins
                err_data_size = len(errors_clean)
                if err_data_size < 200: bins = 30
                elif err_data_size < 2000: bins = 50
                else: bins = 75 # Adjusted default

                # Create histogram
                n, bin_edges, patches = ax_err.hist(errors_clean, bins=bins, density=True, alpha=0.65,
                                            color='#1f77b4', edgecolor='#1f77b4', linewidth=0.5, label='Error Distribution')

                # Add KDE curve
                try:
                    if len(errors_clean) > 1: # Need >1 point for KDE
                        kde = stats.gaussian_kde(errors_clean)
                        kde_x = np.linspace(bin_edges[0], bin_edges[-1], 500)
                        ax_err.plot(kde_x, kde(kde_x), color='#ff7f0e', linewidth=2, label='Density Estimate (KDE)')
                except Exception as kde_err:
                    logging.warning(f"Could not calculate KDE for {target_name}: {kde_err}")

                # Vertical reference lines
                ax_err.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5, label='Zero Error')
                ax_err.axvline(x=mean_error, color='#2ca02c', linestyle='--', linewidth=1.5,
                            label=f'Mean Error ({mean_error:.3f})')
                ax_err.axvline(x=median_error, color='#d62728', linestyle=':', linewidth=1.5,
                            label=f'Median Error ({median_error:.3f})')

                # Add title and labels
                ax_err.set_title(f'Error Distribution Analysis: {ticker} {target_name.title()}', fontsize=14)
                ax_err.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
                ax_err.set_ylabel('Probability Density', fontsize=12)


                # --- POSITIONING & STYLING ---
                # Legend -> Top-Left
                ax_err.legend(loc='upper left', fontsize=10) # Uses default style set by rcParams

                ax_err.grid(True, linestyle='--', alpha=0.5)

                # --- DYNAMIC X-AXIS LIMITS ---
                try:
                    min_err = np.min(errors_clean)
                    max_err = np.max(errors_clean)
                    err_range = max_err - min_err
                    buffer = err_range * 0.10 # 10% buffer

                    if np.isfinite(min_err) and np.isfinite(max_err) and err_range >= 0:
                        x_min_limit = min_err - buffer
                        x_max_limit = max_err + buffer
                        # Ensure zero is visible if the range crosses it, or center if range is zero
                        if err_range < 1e-6 : # Effectively zero range
                             x_min_limit = min_err - 0.5 # Default buffer for zero range
                             x_max_limit = max_err + 0.5
                        else:
                             if x_min_limit > 0 and 0 > min_err - err_range: x_min_limit = min(0, min_err - buffer) # Widen to include 0 if close
                             if x_max_limit < 0 and 0 < max_err + err_range: x_max_limit = max(0, max_err + buffer)

                        ax_err.set_xlim(x_min_limit, x_max_limit)
                    else:
                        logging.warning("Could not determine valid error range for x-axis limits. Using fallback.")
                        # Fallback (optional, could just let matplotlib auto-scale)
                        # error_range_fallback = min(4 * std_error, 1.5 * np.max(np.abs(errors_clean))) if std_error > 0 else np.max(np.abs(errors_clean))
                        # if np.isfinite(error_range_fallback):
                        #    ax_err.set_xlim(mean_error - error_range_fallback, mean_error + error_range_fallback)

                except Exception as xlim_err:
                     logging.warning(f"Error setting error plot x-limits: {xlim_err}")
                # --- END DYNAMIC X-AXIS LIMITS ---

                # Adjust y limit based on histogram or KDE height
                max_y = 0
                if 'kde' in locals() and len(kde_x) > 0:
                    max_y = max(max_y, np.max(kde(kde_x)))
                if len(n) > 0:
                    max_y = max(max_y, np.max(n))
                if max_y > 0:
                    ax_err.set_ylim(0, max_y * 1.1) # 10% headroom


                # Statistics annotation -> Top-Right
                try:
                    skewness = stats.skew(errors_clean)
                    kurt = stats.kurtosis(errors_clean) # Fisher’s definition (normal ==> 0.0)
                except Exception as stat_err:
                    logging.warning(f"Could not calculate skew/kurtosis for {target_name}: {stat_err}")
                    skewness = np.nan
                    kurt = np.nan

                stat_text = [
                    f"Mean: {mean_error:.4f}",
                    f"Median: {median_error:.4f}",
                    f"Std Dev: {std_error:.4f}"
                ]
                if np.isfinite(skewness): stat_text.append(f"Skewness: {skewness:.3f}")
                if np.isfinite(kurt): stat_text.append(f"Kurtosis: {kurt:.3f}")

                ax_err.annotate('\n'.join(stat_text),
                            xy=(0.97, 0.97), # Positioned top-right
                            xycoords='axes fraction',
                            bbox=bbox_style, # Use consistent style
                            fontsize=9,
                            ha='right',
                            va='top')

                # Save the error distribution plot
                err_filename = f"{ticker}_{target_name}_error_dist{plot_suffix}_{timestamp}.png"
                err_plot_path = output_path / err_filename
                plt.tight_layout()
                plt.savefig(err_plot_path, dpi=300, bbox_inches='tight')
                plot_files[f'error_dist_{target_name}'] = str(err_plot_path.resolve())
                logging.info(f"Error distribution plot for {target_name} saved to {err_plot_path}")
                plt.close(fig_err)

        except Exception as e:
            logging.error(f"Failed to generate error distribution plot: {e}", exc_info=True)
            plt.close('all')

    # Reset matplotlib settings potentially changed by context manager
    plt.rcdefaults()

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