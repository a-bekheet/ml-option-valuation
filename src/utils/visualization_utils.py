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
    Generate thesis-quality prediction plots using only matplotlib with full data:
    1. Time series comparison showing prediction accuracy over time (all data points)
    2. 2D histogram showing density of prediction vs. actual values
    3. Enhanced error distribution plots with professional styling

    Args:
        y_true (np.ndarray): True values (original scale recommended).
        y_pred (np.ndarray): Predicted values (original scale recommended).
        target_cols (List[str]): Target column names.
        ticker (str): Stock ticker symbol.
        output_dir (str): Directory path to save plots.
        timestamp (Optional[str]): Timestamp for filenames.
        rmse (Optional[float]): RMSE metric for annotation.
        mae (Optional[float]): MAE metric for annotation.
        n_samples_plot (int): Not used in this version since we plot all data.
        plot_suffix (str): Optional suffix for filenames.

    Returns:
        Dict[str, str]: Dictionary mapping plot types to saved file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default matplotlib style - clean and professional
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'font.size': 11
    })
    
    # Ensure arrays are properly formatted
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

    # Validate target columns
    num_targets = y_true_np.shape[1]
    if len(target_cols) != num_targets:
        logging.warning(f"Target columns ({len(target_cols)}) don't match data dimensions ({num_targets}). Using generic names.")
        target_cols = [f"Target_{i+1}" for i in range(num_targets)]

    plot_files = {}
    data_size = len(y_true_np)
    logging.info(f"Plotting ticker {ticker} with {data_size} data points")
    
    # --- TIME SERIES PLOT ---
    try:
        # Using all indices without sampling
        indices = np.arange(data_size)
        
        # Adjust line and marker properties based on data size for readability
        if data_size <= 500:
            marker_size = 5
            line_width = 1.5
            use_markers = True
        elif data_size <= 5000:
            marker_size = 0
            line_width = 1.2
            use_markers = False
        else:
            marker_size = 0
            line_width = 1.0
            use_markers = False
        
        # Create figure
        fig_ts, axs_ts = plt.subplots(num_targets, 1, figsize=(12, 4 * num_targets), dpi=300, squeeze=False)
        fig_ts.suptitle(f'Temporal Prediction Performance for {ticker} Options', fontsize=16)
            
        for i in range(num_targets):
            target_name = target_cols[i]
            ax = axs_ts[i, 0]
            
            if use_markers:
                ax.plot(indices, y_true_np[:, i], 'o-', label=f'Actual {target_name}', 
                       color='#1f77b4', markersize=marker_size, linewidth=line_width)
                ax.plot(indices, y_pred_np[:, i], 'x--', label=f'Predicted {target_name}', 
                       color='#ff7f0e', markersize=marker_size, linewidth=line_width)
            else:
                ax.plot(indices, y_true_np[:, i], '-', label=f'Actual {target_name}', 
                       color='#1f77b4', linewidth=line_width)
                ax.plot(indices, y_pred_np[:, i], '--', label=f'Predicted {target_name}', 
                       color='#ff7f0e', linewidth=line_width)
            
            # Add error shading for smaller datasets
            if data_size <= 10000:
                ax.fill_between(indices, y_true_np[:, i], y_pred_np[:, i], 
                               color='#ff7f0e', alpha=0.15)
            
            ax.set_title(f'Sequential {target_name.title()} Prediction Analysis', fontsize=14)
            ax.set_xlabel('Sample Index (Chronological Sequence)', fontsize=12)
            ax.set_ylabel(f'{target_name.title()} Value ($)', fontsize=12)
            ax.legend(fontsize=12, loc='best')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add metrics annotation if available
            if rmse is not None or mae is not None:
                metric_text = []
                if rmse is not None and not np.isnan(rmse): 
                    metric_text.append(f'RMSE: {rmse:.4f}')
                if mae is not None and not np.isnan(mae): 
                    metric_text.append(f'MAE: {mae:.4f}')
                    
                if metric_text:
                    ax.annotate('\n'.join(metric_text), xy=(0.02, 0.96), xycoords='axes fraction',
                              bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.8),
                              fontsize=11, ha='left', va='top')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        ts_filename = f"{ticker}_timeseries{plot_suffix}_{timestamp}.png"
        ts_plot_path = output_path / ts_filename
        plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
        plot_files['time_series'] = str(ts_plot_path.resolve())
        logging.info(f"Time series plot saved to {ts_plot_path}")
        plt.close(fig_ts)
    except Exception as e:
        logging.error(f"Failed to generate time series plot: {e}")
        import traceback
        logging.error(traceback.format_exc())
        plt.close('all')
    
    # --- 2D HISTOGRAM PREDICTION ANALYSIS ---
    try:
        # Create one figure per target for 2D histogram
        for i in range(num_targets):
            target_name = target_cols[i]
            true_values = y_true_np[:, i]
            pred_values = y_pred_np[:, i]
            
            # Create figure
            fig_hist = plt.figure(figsize=(8, 7), dpi=300)
            ax_hist = fig_hist.add_subplot(111)
            
            # Set up the title
            ax_hist.set_title(f'Prediction Accuracy Distribution for {ticker} {target_name.title()}', fontsize=14)
            
            # Determine appropriate bin count based on data size
            if data_size < 500:
                bins = 25
            elif data_size < 5000:
                bins = 50
            else:
                bins = 100  # Increased bins for larger datasets
            
            # Determine data range with some padding
            min_val = min(np.nanmin(true_values), np.nanmin(pred_values))
            max_val = max(np.nanmax(true_values), np.nanmax(pred_values))
            data_range = max_val - min_val
            buffer = data_range * 0.05
            plot_range = [[min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer]]
            
            # Create 2D histogram with blue-to-red colormap
            h = ax_hist.hist2d(true_values, pred_values, 
                            bins=bins, 
                            range=plot_range,
                            cmap='viridis',  # Professional coloring
                            norm=plt.cm.colors.LogNorm(),
                            cmin=1)
            
            fig_hist.colorbar(h[3], ax=ax_hist, label='Count (log scale)')
            
            # Add perfect prediction line
            ax_hist.plot(plot_range[0], plot_range[0], 'k--', linewidth=1.5, label='Perfect Prediction')
            
            # Add regression line
            try:
                valid_mask = ~np.isnan(true_values) & ~np.isnan(pred_values)
                if np.sum(valid_mask) > 2:
                    z = np.polyfit(true_values[valid_mask], pred_values[valid_mask], 1)
                    p = np.poly1d(z)
                    ax_hist.plot(plot_range[0], p(plot_range[0]), 'r-', linewidth=1.5, 
                            label=f'Regression Line (slope={z[0]:.3f})')
            except:
                pass
            
            # Add legend in upper left
            ax_hist.legend(loc='upper left', fontsize=10, 
                        framealpha=0.9, facecolor='white', edgecolor='gray')
            
            # Add labels
            ax_hist.set_xlabel(f'Actual {target_name} ($)', fontsize=12)
            ax_hist.set_ylabel(f'Predicted {target_name} ($)', fontsize=12)
            
            # Add metrics annotation box on the right side to avoid overlap with legend
            if rmse is not None or mae is not None:
                metric_text = []
                if rmse is not None and not np.isnan(rmse): 
                    metric_text.append(f'RMSE: {rmse:.4f}')
                if mae is not None and not np.isnan(mae): 
                    metric_text.append(f'MAE: {mae:.4f}')
                
                # Calculate R² if enough points
                if len(true_values) > 3:
                    try:
                        valid_mask = ~np.isnan(true_values) & ~np.isnan(pred_values)
                        r2 = np.corrcoef(true_values[valid_mask], pred_values[valid_mask])[0, 1]**2
                        metric_text.append(f'R²: {r2:.4f}')
                    except:
                        pass
                        
                if metric_text:
                    # Position on the right side
                    ax_hist.annotate('\n'.join(metric_text), 
                                xy=(0.95, 0.95), 
                                xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.5", 
                                        facecolor='white', 
                                        alpha=0.9,
                                        edgecolor='gray'),
                                fontsize=10, 
                                ha='right', 
                                va='top')
            
            # Set equal aspect ratio with square plot
            ax_hist.set_aspect('equal')
            ax_hist.grid(True, linestyle='--', alpha=0.5)
            
            # Save the histogram
            hist_filename = f"{ticker}_{target_name}_2dhist{plot_suffix}_{timestamp}.png"
            hist_plot_path = output_path / hist_filename
            plt.tight_layout()
            plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
            plot_files[f'hist2d_{target_name}'] = str(hist_plot_path.resolve())
            logging.info(f"2D histogram for {target_name} saved to {hist_plot_path}")
            plt.close(fig_hist)
            
    except Exception as e:
        logging.error(f"Failed to generate 2D histogram plot: {e}")
        import traceback
        logging.error(traceback.format_exc())
        plt.close('all')
    
    # --- ERROR DISTRIBUTION PLOTS (SEPARATE) ---
    try:
        for i in range(num_targets):
            target_name = target_cols[i]
            true_values = y_true_np[:, i]
            pred_values = y_pred_np[:, i]

            # Calculate errors
            errors = pred_values - true_values

            # Create separate figure for error distribution
            fig_err = plt.figure(figsize=(10, 6), dpi=300)
            ax_err = fig_err.add_subplot(111)

            # Calculate error statistics
            mean_error = np.nanmean(errors)
            median_error = np.nanmedian(errors)
            std_error = np.nanstd(errors)

            # Calculate number of bins - significantly increased for smoother distribution
            if data_size < 200:
                bins = 30
            elif data_size < 2000:
                bins = 60
            else:
                bins = 100

            # Create histogram with enhanced styling using a professional color scheme
            n, bins, patches = ax_err.hist(errors, bins=bins, density=True, alpha=0.7, 
                                        color='#1f77b4', edgecolor='#1f77b4', linewidth=0.5)

            # Calculate KDE curve manually with matplotlib
            try:
                from scipy import stats
                # Use Scott's rule for smoothing
                kde_x = np.linspace(np.nanmin(errors), np.nanmax(errors), 1000)
                kde = stats.gaussian_kde(errors[~np.isnan(errors)])
                ax_err.plot(kde_x, kde(kde_x), color='#ff7f0e', linewidth=2.5, label='Density Estimate')
            except:
                # If scipy unavailable, skip KDE curve
                pass

            # Add vertical reference lines
            ax_err.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Zero Error')
            ax_err.axvline(x=mean_error, color='#2ca02c', linestyle='--', linewidth=2.5, 
                        label=f'Mean Error: {mean_error:.4f}')
            ax_err.axvline(x=median_error, color='#d62728', linestyle=':', linewidth=2.5, 
                        label=f'Median Error: {median_error:.4f}')

            # Add standard deviation lines
            ax_err.axvline(x=mean_error + std_error, color='#9467bd', linestyle='-.', alpha=0.8, linewidth=2,
                        label=f'±1 Std Dev: {std_error:.4f}')
            ax_err.axvline(x=mean_error - std_error, color='#9467bd', linestyle='-.', alpha=0.8, linewidth=2)

            # Add title and labels - more thesis-appropriate
            ax_err.set_title(f'Error Distribution Analysis: {ticker} {target_name.title()} Prediction Model', fontsize=14)
            ax_err.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
            ax_err.set_ylabel('Probability Density', fontsize=12)

            # Add legend with better placement and styling
            ax_err.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10, framealpha=0.9)

            # Add gridlines and set x limits to reasonable range
            ax_err.grid(True, linestyle='--', alpha=0.5)

            # Set x limits to capture the main distribution (±4σ or the actual range if smaller)
            error_range = min(4 * std_error, 1.5 * np.max(np.abs(errors)))
            ax_err.set_xlim(mean_error - error_range, mean_error + error_range)

            # Set y limit to capture full density plus some headroom
            if len(errors) > 0:
                try:
                    # Calculate y-limit based on histogram heights rather than KDE
                    max_bin_height = max(n) * 1.1  # 10% headroom
                    ax_err.set_ylim(0, max_bin_height)
                except:
                    pass

            # Add text box with key statistical insights
            try:
                from scipy import stats
                skewness = stats.skew(errors[~np.isnan(errors)])
                kurtosis = stats.kurtosis(errors[~np.isnan(errors)])
            except:
                # If scipy unavailable, use numpy or simply skip these metrics
                try:
                    skewness = np.nanmean(((errors - mean_error) / std_error) ** 3)
                    kurtosis = np.nanmean(((errors - mean_error) / std_error) ** 4) - 3
                except:
                    skewness = np.nan
                    kurtosis = np.nan

            stat_text = [
                f"Mean Error: {mean_error:.4f}",
                f"Median Error: {median_error:.4f}",
                f"Std Dev: {std_error:.4f}"
            ]

            if not np.isnan(skewness):
                stat_text.append(f"Skewness: {skewness:.4f}")
            if not np.isnan(kurtosis):
                stat_text.append(f"Kurtosis: {kurtosis:.4f}")

            ax_err.annotate('\n'.join(stat_text), xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                        fontsize=9, ha='left', va='top')

            # Save the error distribution plot
            err_filename = f"{ticker}_{target_name}_error_dist{plot_suffix}_{timestamp}.png"
            err_plot_path = output_path / err_filename
            plt.tight_layout()
            plt.savefig(err_plot_path, dpi=300, bbox_inches='tight')
            plot_files[f'error_dist_{target_name}'] = str(err_plot_path.resolve())
            logging.info(f"Error distribution plot for {target_name} saved to {err_plot_path}")
            plt.close(fig_err)
            
    except Exception as e:
        logging.error(f"Failed to generate error distribution plot: {e}")
        import traceback
        logging.error(traceback.format_exc())
        plt.close('all')
    
    # Reset matplotlib settings
    plt.style.use('default')
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