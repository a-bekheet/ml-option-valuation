import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch

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
        