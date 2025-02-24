import torch
import torch.nn as nn
import os
from datetime import datetime
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_errors(y_true, y_pred):
    """Calculate various error metrics between predicted and actual values."""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Calculate errors (averaged over both targets)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def analyze_model_architecture(model, input_size=23, seq_len=15, batch_size=32):
    """Analyze the architecture of the model, including parameter count and tensor shapes."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    layer_shapes = {}
    
    def hook_fn(module, input, output, name):
        def get_tensor_shape(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            elif isinstance(x, tuple):
                return tuple(get_tensor_shape(t) for t in x if isinstance(t, torch.Tensor))
            return None

        layer_shapes[name] = {
            'input_shape': [tuple(i.shape) for i in input],
            'output_shape': get_tensor_shape(output)
        }
    
    hooks = []
    for name, layer in model.named_children():
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, name=name: hook_fn(m, i, o, name)
        ))
    
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layer_shapes': layer_shapes
    }

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    def log_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    early_stopping = EarlyStopping(patience=5)
    
    model.to(device)
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(epochs):
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
        
        scheduler.step(avg_val_loss)
        current_lr = log_lr(optimizer)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break
    
    return train_losses, val_losses

def list_available_models(models_dir):
    """List and return available trained models."""
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        print("\nNo saved models found in", models_dir)
        return None
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    return model_files

def select_model(model_files):
    """Let user select a model from the list."""
    while True:
        try:
            model_choice = int(input("\nSelect a model number: "))
            if 1 <= model_choice <= len(model_files):
                return model_files[model_choice-1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nModel selection cancelled")
            return None

def get_model_class_from_name(model_name, HybridRNNModel, GRUGRUModel, LSTMLSTMModel):
    """Determine the model class based on the model name."""
    model_name = model_name.lower()
    if "gru-gru" in model_name:
        return GRUGRUModel, "GRU-GRU"
    elif "lstm-lstm" in model_name:
        return LSTMLSTMModel, "LSTM-LSTM"
    else:
        return HybridRNNModel, "LSTM-GRU"
        
def load_model(model_path, model_class, input_size, 
               hidden_size_lstm=128, hidden_size_gru=128, 
               hidden_size_lstm1=128, hidden_size_lstm2=128,
               hidden_size_gru1=128, hidden_size_gru2=128,
               num_layers=2, output_size=1):
    """
    Load a saved model with proper architecture class.
    Supports all three architecture types.
    """
    # Import needed here to avoid circular import
    from nn import HybridRNNModel, GRUGRUModel, LSTMLSTMModel
    
    if model_class == HybridRNNModel:
        model = model_class(
            input_size=input_size,
            hidden_size_lstm=hidden_size_lstm,
            hidden_size_gru=hidden_size_gru,
            num_layers=num_layers,
            output_size=output_size
        )
    elif model_class == GRUGRUModel:
        model = model_class(
            input_size=input_size,
            hidden_size_gru1=hidden_size_gru1,
            hidden_size_gru2=hidden_size_gru2,
            num_layers=num_layers,
            output_size=output_size
        )
    elif model_class == LSTMLSTMModel:
        model = model_class(
            input_size=input_size,
            hidden_size_lstm1=hidden_size_lstm1,
            hidden_size_lstm2=hidden_size_lstm2,
            num_layers=num_layers,
            output_size=output_size
        )
    
    model.load_state_dict(torch.load(model_path))
    return model

def run_existing_model(model_path, model_class, dataset, target_cols=["bid", "ask"]):
    """Load and run predictions with an existing model."""
    model = load_model(
        model_path, 
        model_class,
        input_size=dataset.n_features, 
        output_size=len(dataset.target_cols)
    )
    model.eval()
    
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"\nMaking predictions for {dataset.ticker}...")
    with torch.no_grad():
        for x_seq, y_val in data_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            y_pred = model(x_seq)
            all_y_true.extend(y_val.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())
    
    errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
    print("\nPrediction Metrics:")
    print("-" * 50)
    print(f"MSE: {errors['mse']:.6f}")
    print(f"RMSE: {errors['rmse']:.6f}")
    print(f"MAE: {errors['mae']:.6f}")
    print(f"MAPE: {errors['mape']:.2f}%")

# Handler Functions moved from nn.py
def handle_train_model(config, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, save_and_display_results, extended_train_model_with_tracking, get_available_tickers, select_ticker, StockOptionDataset):
    """Handle the model training workflow."""
    try:
        logging.info("Starting model training...")
        # Ask user if they want to use performance tracking
        use_tracking = input("\nUse performance tracking? (y/n): ").lower().startswith('y')
        
        # Ask user to select model architecture
        print("\nSelect model architecture:")
        print("1. LSTM-GRU Hybrid (default)")
        print("2. GRU-GRU")
        print("3. LSTM-LSTM")
        
        arch_choice = input("\nEnter choice (1-3): ").strip()
        if arch_choice == "2":
            architecture_type = "GRU-GRU"
        elif arch_choice == "3":
            architecture_type = "LSTM-LSTM"
        else:
            architecture_type = "LSTM-GRU"
        
        # Get available tickers and select one
        tickers, counts = get_available_tickers(config['data_dir'])
        ticker = select_ticker(tickers, counts)
        
        # Initialize dataset
        dataset = StockOptionDataset(
            data_dir=config['data_dir'], 
            ticker=ticker, 
            seq_len=config['seq_len'], 
            target_cols=config['target_cols']
        )
        
        if len(dataset) < 1:
            raise ValueError("Insufficient data for sequence creation!")
        
        # Split dataset maintaining temporal order
        total_len = len(dataset)
        train_len = int(0.80 * total_len)
        val_len = int(0.10 * total_len)
        
        indices = list(range(total_len))
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len+val_len]
        test_indices = indices[train_len+val_len:]
        
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model based on selected architecture
        if architecture_type == "GRU-GRU":
            model = GRUGRUModel(
                input_size=dataset.n_features,
                hidden_size_gru1=config['hidden_size_gru'],
                hidden_size_gru2=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print(f"\nInitialized GRU-GRU architecture")
        elif architecture_type == "LSTM-LSTM":
            model = LSTMLSTMModel(
                input_size=dataset.n_features,
                hidden_size_lstm1=config['hidden_size_lstm'],
                hidden_size_lstm2=config['hidden_size_lstm'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print(f"\nInitialized LSTM-LSTM architecture")
        else:  # Default to LSTM-GRU hybrid
            model = HybridRNNModel(
                input_size=dataset.n_features,
                hidden_size_lstm=config['hidden_size_lstm'],
                hidden_size_gru=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print(f"\nInitialized LSTM-GRU hybrid architecture")
        
        model_analysis = analyze_model_architecture(
            model, 
            input_size=dataset.n_features,
            seq_len=config['seq_len']
        )
        
        if use_tracking:
            # Use performance tracking training
            model, history, log_path = extended_train_model_with_tracking(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=config['epochs'],
                lr=1e-3,
                device=device,
                ticker=ticker,
                architecture_name=architecture_type,
                target_cols=config['target_cols']
            )
            print(f"\nPerformance log saved to: {log_path}")
        else:
            # Standard training without performance tracking
            train_losses, val_losses = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config['epochs'],
                lr=1e-3,
                device=device
            )
            
            history = {'train_losses': train_losses, 'val_losses': val_losses}
            
            # Final evaluation
            model.eval()
            test_loss = 0.0
            all_y_true = []
            all_y_pred = []
            criterion = nn.MSELoss()
            
            with torch.no_grad():
                for x_seq, y_val in test_loader:
                    x_seq, y_val = x_seq.to(device), y_val.to(device)
                    y_pred = model(x_seq)
                    loss = criterion(y_pred, y_val)
                    test_loss += loss.item()
                    
                    all_y_true.extend(y_val.cpu().numpy())
                    all_y_pred.extend(y_pred.cpu().numpy())
            
            errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
            print("\nFinal Test Set Metrics:")
            print("-" * 50)
            print(f"MSE: {errors['mse']:.6f}")
            print(f"RMSE: {errors['rmse']:.6f}")
            print(f"MAE: {errors['mae']:.6f}")
            print(f"MAPE: {errors['mape']:.2f}%")
            
        save_and_display_results(model, history, model_analysis, ticker, config['target_cols'], models_dir=config['models_dir'])
        logging.info("Model training completed successfully")
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        print(f"\nError: {str(e)}")

def handle_run_model(config, models_dir, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, get_available_tickers, select_ticker, StockOptionDataset):
    """Handle the model prediction workflow."""
    try:
        model_files = list_available_models(models_dir)
        if not model_files:
            return

        selected_model = select_model(model_files)
        if not selected_model:
            return

        model_path = os.path.join(models_dir, selected_model)
        
        # Determine model architecture based on filename
        model_class, architecture_name = get_model_class_from_name(
            selected_model, 
            HybridRNNModel, 
            GRUGRUModel, 
            LSTMLSTMModel
        )
        print(f"\nDetected {architecture_name} architecture")
        
        # Get available tickers and select one
        tickers, counts = get_available_tickers(config['data_dir'])
        ticker = select_ticker(tickers, counts)
        
        # Create dataset for the selected ticker
        dataset = StockOptionDataset(
            data_dir=config['data_dir'],
            ticker=ticker,
            target_cols=config['target_cols']
        )
        
        logging.info(f"Running predictions with model: {selected_model}")
        run_existing_model(
            model_path,
            model_class,
            dataset,
            target_cols=config['target_cols']
        )
        logging.info("Predictions completed successfully")
    except Exception as e:
        logging.error(f"Error during model prediction: {str(e)}")
        print(f"\nError: {str(e)}")

def handle_analyze_architecture(config, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, display_model_analysis):
    """Handle the model architecture analysis workflow."""
    try:
        # Allow user to select architecture to analyze
        print("\nSelect model architecture to analyze:")
        print("1. LSTM-GRU Hybrid (default)")
        print("2. GRU-GRU")
        print("3. LSTM-LSTM")
        
        arch_choice = input("\nEnter choice (1-3): ").strip()
        
        if arch_choice == "2":
            model = GRUGRUModel(
                input_size=23,
                hidden_size_gru1=config['hidden_size_gru'],
                hidden_size_gru2=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print("\nAnalyzing GRU-GRU architecture...")
        elif arch_choice == "3":
            model = LSTMLSTMModel(
                input_size=23,
                hidden_size_lstm1=config['hidden_size_lstm'],
                hidden_size_lstm2=config['hidden_size_lstm'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print("\nAnalyzing LSTM-LSTM architecture...")
        else:
            model = HybridRNNModel(
                input_size=23,
                hidden_size_lstm=config['hidden_size_lstm'],
                hidden_size_gru=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            print("\nAnalyzing LSTM-GRU hybrid architecture...")
                
        analysis = analyze_model_architecture(model)
        display_model_analysis(analysis)
        logging.info("Architecture analysis completed")
    except Exception as e:
        logging.error(f"Error during architecture analysis: {str(e)}")
        print(f"\nError: {str(e)}")

def handle_benchmark_architectures(config, HybridRNNModel, GRUGRUModel, LSTMLSTMModel, 
                                  benchmark_architectures, generate_architecture_comparison, 
                                  visualize_architectures, get_available_tickers, 
                                  select_ticker, StockOptionDataset):
    """Handle architecture benchmarking workflow."""
    try:
        logging.info("Starting architecture benchmarking...")
        print("\nBenchmarking Different RNN Architectures")
        print("-" * 50)
        
        # Get ticker
        tickers, counts = get_available_tickers(config['data_dir'])
        ticker = select_ticker(tickers, counts)
        
        # Initialize dataset
        dataset = StockOptionDataset(
            data_dir=config['data_dir'], 
            ticker=ticker, 
            seq_len=config['seq_len'], 
            target_cols=config['target_cols']
        )
        
        if len(dataset) < 1:
            raise ValueError("Insufficient data for sequence creation!")
            
        # Split dataset
        total_len = len(dataset)
        train_len = int(0.80 * total_len)
        val_len = int(0.10 * total_len)
        test_len = total_len - train_len - val_len
        
        indices = list(range(total_len))
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len+val_len]
        test_indices = indices[train_len+val_len:]
        
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices)
        
        batch_size = int(input("\nEnter batch size (default: 32): ") or "32")
        epochs = int(input("Enter number of epochs (default: 20): ") or "20")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        # Create the models to benchmark
        hidden_size = config['hidden_size_lstm']  # Use same hidden size for fair comparison
        num_layers = config['num_layers']
        input_size = dataset.n_features
        output_size = len(config['target_cols'])
        
        lstm_gru_model = HybridRNNModel(
            input_size=input_size,
            hidden_size_lstm=hidden_size,
            hidden_size_gru=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        )
        
        gru_gru_model = GRUGRUModel(
            input_size=input_size,
            hidden_size_gru1=hidden_size,
            hidden_size_gru2=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        )
        
        lstm_lstm_model = LSTMLSTMModel(
            input_size=input_size,
            hidden_size_lstm1=hidden_size,
            hidden_size_lstm2=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        )
        
        # Prepare model configurations
        models = [
            {'name': 'LSTM-GRU Hybrid', 'model': lstm_gru_model},
            {'name': 'GRU-GRU', 'model': gru_gru_model},
            {'name': 'LSTM-LSTM', 'model': lstm_lstm_model}
        ]
        
        # Run benchmarks
        print("\nStarting benchmark of 3 different architectures...")
        print("This may take some time. A detailed log will be created for each architecture.")
        
        log_paths = benchmark_architectures(
            models=models,
            data_loaders=data_loaders,
            epochs=epochs,
            ticker=ticker,
            target_cols=config['target_cols'],
            save_dir=config['performance_logs_dir']
        )
        
        # Generate comparison summary
        summary_path = generate_architecture_comparison(log_paths)
        print(f"\nArchitecture comparison saved to: {summary_path}")
        
        # Create visualizations
        viz_paths = visualize_architectures(log_paths, output_dir=config['performance_logs_dir'])
        if viz_paths:
            print(f"\nVisualization charts created:")
            for path in viz_paths:
                print(f"- {path}")
                
        logging.info("Architecture benchmarking completed successfully")
        
    except Exception as e:
        logging.error(f"Error during architecture benchmarking: {str(e)}")
        print(f"\nError: {str(e)}")