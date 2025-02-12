import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datetime import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

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

def get_available_tickers(data_dir):
    """Load and return sorted list of available tickers from metadata file."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"\nError: Data directory '{data_dir}' not found!\n"
            f"Please run the split_data.py script first to prepare the data:\n"
            f"  python split_data.py"
        )
        
    metadata_file = os.path.join(data_dir, 'ticker_metadata.csv')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"\nError: Metadata file not found at '{metadata_file}'!\n"
            f"Please run the split_data.py script first to prepare the data:\n"
            f"  python split_data.py"
        )
        
    metadata = pd.read_csv(metadata_file)
    return list(metadata['ticker']), list(metadata['count'])

def select_ticker(tickers, counts):
    """Display tickers with their data counts and get user selection."""
    print("\nAvailable tickers and their data points:")
    print("-" * 50)
    print(f"{'#':<4} {'Ticker':<10} {'Data Points':>12}")
    print("-" * 50)
    
    for i, (ticker, count) in enumerate(zip(tickers, counts), 1):
        print(f"{i:<4} {ticker:<10} {count:>12,}")
    
    while True:
        try:
            choice = input("\nEnter the number of the ticker you want to analyze (or 'q' to quit): ")
            if choice.lower() == 'q':
                exit()
            choice = int(choice)
            if 1 <= choice <= len(tickers):
                selected_ticker = tickers[choice-1]
                selected_count = counts[choice-1]
                print(f"\nSelected ticker: {selected_ticker} ({selected_count:,} data points)")
                return selected_ticker
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

class StockOptionDataset(Dataset):
    def __init__(self, data_dir, ticker, seq_len=15, target_cols=["bid", "ask"]):
        """
        Loads data for a given ticker from its specific file.
        """
        file_path = os.path.join(data_dir, f"option_data_scaled_{ticker}.csv")
        print(f"Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No data file found for ticker {ticker}")
            
        df = pd.read_csv(file_path)
        
        # Define the feature columns (excluding target columns)
        feature_cols = [
            "strike", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume",
            "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d",
            "day_of_week", "day_of_month", "day_of_year"
        ]
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.ticker = ticker

        # Ensure that none of the required columns are missing
        df.dropna(subset=feature_cols + target_cols, inplace=True)
        
        # Concatenate features and targets
        data_np = df[feature_cols + target_cols].to_numpy(dtype=np.float32)
        
        self.data = data_np
        self.n_features = len(feature_cols)
        self.n_targets = len(target_cols)
        self.seq_len = seq_len
        self.n_samples = self.data.shape[0]
        self.max_index = self.n_samples - self.seq_len - 1
        
    def __len__(self):
        return max(0, self.max_index)
    
    def __getitem__(self, idx):
        x_seq = self.data[idx : idx + self.seq_len, :self.n_features]
        y_val = self.data[idx + self.seq_len, self.n_features : self.n_features + self.n_targets]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

# [The ImprovedMixedRNNModel class remains exactly the same as in the original code]
class ImprovedMixedRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size_lstm=64, hidden_size_gru=64, num_layers=2, output_size=1):
        super(ImprovedMixedRNNModel, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.mid_bn = nn.BatchNorm1d(hidden_size_lstm)
        self.dropout = nn.Dropout(0.2)
        
        self.gru = nn.GRU(
            input_size=hidden_size_lstm,
            hidden_size=hidden_size_gru,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.bn_final = nn.BatchNorm1d(hidden_size_gru)
        self.fc1 = nn.Linear(hidden_size_gru, hidden_size_gru // 2)
        self.fc2 = nn.Linear(hidden_size_gru // 2, output_size)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x = x.view(-1, features)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, features)
        
        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out.contiguous()
        batch_size, seq_len, hidden_size = lstm_out.size()
        lstm_out = lstm_out.view(-1, hidden_size)
        lstm_out = self.mid_bn(lstm_out)
        lstm_out = lstm_out.view(batch_size, seq_len, hidden_size)
        lstm_out = self.dropout(lstm_out)
        
        gru_out, _ = self.gru(lstm_out)
        
        final_out = gru_out[:, -1, :]
        final_out = self.bn_final(final_out)
        final_out = self.dropout(final_out)
        final_out = torch.relu(self.fc1(final_out))
        final_out = self.fc2(final_out)
        
        return final_out

# [EarlyStopping, calculate_errors, and analyze_model_architecture functions remain the same]

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

def train_option_model(data_dir, ticker=None, seq_len=15, batch_size=128, epochs=20, 
                      hidden_size_lstm=128, hidden_size_gru=128, num_layers=2,
                      target_cols=["bid", "ask"]):
    """
    Train the option pricing model using ticker-specific data files.
    """
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
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            for x_seq, y_val in train_loader:
                x_seq, y_val = x_seq.to(device), y_val.to(device)
                
                optimizer.zero_grad()
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_val)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x_seq, y_val in val_loader:
                    x_seq, y_val = x_seq.to(device), y_val.to(device)
                    y_pred = model(x_seq)
                    loss = criterion(y_pred, y_val)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            current_lr = log_lr(optimizer)
            
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
            
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        return train_losses, val_losses
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get available tickers and select one if not provided
    if ticker is None:
        tickers, counts = get_available_tickers(data_dir)
        ticker = select_ticker(tickers, counts)
    
    # Initialize dataset
    dataset = StockOptionDataset(data_dir=data_dir, ticker=ticker, seq_len=seq_len, target_cols=target_cols)
    
    if len(dataset) < 1:
        raise ValueError("Insufficient data for sequence creation!")
    
    # Split dataset maintaining temporal order
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
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize and train model
    model = ImprovedMixedRNNModel(
        input_size=dataset.n_features,
        hidden_size_lstm=hidden_size_lstm,
        hidden_size_gru=hidden_size_gru,
        num_layers=num_layers,
        output_size=len(target_cols)
    )
    
    model_analysis = analyze_model_architecture(
        model, 
        input_size=dataset.n_features,
        seq_len=seq_len
    )
    
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device
    )
    
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
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}, model_analysis, dataset.ticker, target_cols

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

def run_existing_model(model_path, data_dir, ticker=None, target_cols=["bid", "ask"]):
    """
    Load and run predictions with an existing model on ticker-specific data.
    """
    if ticker is None:
        tickers, counts = get_available_tickers(data_dir)
        ticker = select_ticker(tickers, counts)
    
    dataset = StockOptionDataset(data_dir=data_dir, ticker=ticker, target_cols=target_cols)
    
    model = ImprovedMixedRNNModel(
        input_size=dataset.n_features,
        hidden_size_lstm=128,
        hidden_size_gru=128,
        num_layers=2,
        output_size=len(target_cols)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"\nMaking predictions for {ticker}...")
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

def display_menu():
    """Display the main menu and get user choice."""
    print("\nOption Trading Model - Main Menu")
    print("-" * 50)
    print("1. Train new model")
    print("2. Run existing model")
    print("3. Analyze network architecture")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ")
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            print("Invalid choice. Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def analyze_model_architecture(model, input_size=23, seq_len=15, batch_size=32):
    """
    Analyze the architecture of the model, including parameter count and tensor shapes.
    """
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

def main():
    config = {
        'data_dir': "data_files/split_data",  # Directory containing ticker-specific files
        'seq_len': 15,
        'batch_size': 32,
        'epochs': 20,
        'hidden_size_lstm': 128,
        'hidden_size_gru': 128,
        'num_layers': 2,
        'ticker': None,  # if None, user will be prompted
        'target_cols': ["bid", "ask"]
    }
    
    while True:
        choice = display_menu()
        
        if choice == 1:
            print("\nTraining new model...")
            try:
                model, history, analysis, ticker, target_cols = train_option_model(**config)
            except FileNotFoundError as e:
                print(e)
                input("\nPress Enter to return to main menu...")
                continue
            save_and_display_results(model, history, analysis, ticker, target_cols)
            
        elif choice == 2:
            models_dir = "models"
            if not os.path.exists(models_dir):
                print("\nNo models directory found. Please train a model first.")
                continue
                
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            
            if not model_files:
                print("\nNo saved models found in", models_dir)
                continue
            
            print("\nAvailable models:")
            for i, model_file in enumerate(model_files, 1):
                print(f"{i}. {model_file}")
            
            while True:
                try:
                    model_choice = int(input("\nSelect a model number: "))
                    if 1 <= model_choice <= len(model_files):
                        model_path = os.path.join(models_dir, model_files[model_choice-1])
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            run_existing_model(model_path, config['data_dir'], ticker=config['ticker'], target_cols=config['target_cols'])
            
        elif choice == 3:
            print("\nAnalyzing network architecture...")
            model = ImprovedMixedRNNModel(
                input_size=23,
                hidden_size_lstm=config['hidden_size_lstm'],
                hidden_size_gru=config['hidden_size_gru'],
                num_layers=config['num_layers'],
                output_size=len(config['target_cols'])
            )
            analysis = analyze_model_architecture(model)
            
            print("\nNetwork Architecture Analysis:")
            print("-" * 50)
            print(f"Total parameters: {analysis['total_parameters']:,}")
            print(f"Trainable parameters: {analysis['trainable_parameters']:,}")
            print("\nLayer Shapes:")
            for layer_name, shapes in analysis['layer_shapes'].items():
                print(f"\n{layer_name}:")
                print(f"  Input shape: {shapes['input_shape']}")
                print(f"  Output shape: {shapes['output_shape']}")
                
        elif choice == 4:
            print("\nExiting program...")
            break
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()