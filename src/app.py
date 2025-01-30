import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
import os

# -----------------------------
# BACKEND CODE (Original Logic)
# -----------------------------

def get_available_tickers(csv_file):
    """Load and return sorted list of unique tickers with their data counts."""
    df = pd.read_csv(csv_file)
    ticker_counts = df['ticker'].value_counts().sort_index()
    return list(ticker_counts.index), list(ticker_counts.values)

class StockOptionDataset(Dataset):
    def __init__(self, csv_file, ticker="CGC", seq_len=15):
        super().__init__()
        df = pd.read_csv(csv_file)

        # Filter data for specified ticker
        df = df[df['ticker'] == ticker].copy()
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        feature_cols = [
            "strike", "bid", "ask", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume",
            "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d",
            "day_of_week", "day_of_month", "day_of_year"
        ]

        target_col = "lastPrice"
        df.dropna(subset=feature_cols + [target_col], inplace=True)
        data_np = df[feature_cols + [target_col]].to_numpy(dtype=np.float32)

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_features = len(feature_cols)
        self.seq_len = seq_len
        self.data = data_np
        self.n_samples = self.data.shape[0]
        self.max_index = self.n_samples - self.seq_len - 1

    def __len__(self):
        # Avoid negative length if data is too small
        return max(0, self.max_index)

    def __getitem__(self, idx):
        x_seq = self.data[idx : idx + self.seq_len, :self.n_features]
        y_val = self.data[idx + self.seq_len, self.n_features]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

class ImprovedMixedRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size_lstm=128, hidden_size_gru=128, num_layers=2):
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
        self.fc2 = nn.Linear(hidden_size_gru // 2, 1)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        # Apply batch normalization across features
        x = x.view(-1, features)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, features)
        
        lstm_out, _ = self.lstm(x)
        
        # Flatten, batchnorm, un-flatten
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
    
    # Calculate errors
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    # Avoid division by zero in MAPE
    non_zero_indices = (y_true != 0)
    if np.any(non_zero_indices):
        mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    else:
        mape = np.nan

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def analyze_model_architecture(model, input_size=23, seq_len=15, batch_size=32):
    """
    Analyze the architecture of the model, including parameter count and tensor shapes.
    
    Args:
        model: The PyTorch model to analyze
        input_size: Number of input features
        seq_len: Length of input sequence
        batch_size: Batch size for shape analysis
    
    Returns:
        dict: Dictionary containing model statistics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create dummy input for shape analysis
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    
    # Dictionary to store shapes
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
    
    # Register hooks for each layer
    hooks = []
    for name, layer in model.named_children():
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, name=name: hook_fn(m, i, o, name)
        ))
    
    # Forward pass with dummy input
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Remove hooks
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
        # Training phase
        model.train()
        total_train_loss = 0.0
        for x_seq, y_val in train_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_seq)
            loss = criterion(y_pred.squeeze(), y_val)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_seq, y_val in val_loader:
                x_seq, y_val = x_seq.to(device), y_val.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred.squeeze(), y_val)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = log_lr(optimizer)
        
        # Check early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            break
    
    return train_losses, val_losses

def train_option_model(data_path, ticker, seq_len=15, batch_size=128, epochs=20, 
                      hidden_size_lstm=128, hidden_size_gru=128, num_layers=2):
    """
    Train the option pricing model with the specified parameters.
    
    Args:
        data_path: Path to the CSV data file
        ticker: Stock ticker to analyze
        seq_len: Length of input sequence
        batch_size: Training batch size
        epochs: Number of training epochs
        hidden_size_lstm: Hidden size for LSTM layer
        hidden_size_gru: Hidden size for GRU layer
        num_layers: Number of layers in both LSTM and GRU
    
    Returns:
        tuple: (trained_model, training_history, model_analysis, final_test_errors)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize dataset
    dataset = StockOptionDataset(csv_file=data_path, ticker=ticker, seq_len=seq_len)
    
    if len(dataset) < 1:
        raise ValueError("Insufficient data for sequence creation!")
    
    # Data splitting
    total_len = len(dataset)
    train_len = int(0.70 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ImprovedMixedRNNModel(
        input_size=dataset.n_features,
        hidden_size_lstm=hidden_size_lstm,
        hidden_size_gru=hidden_size_gru,
        num_layers=num_layers
    )
    
    # Analyze model architecture
    model_analysis = analyze_model_architecture(
        model, 
        input_size=dataset.n_features,
        seq_len=seq_len
    )
    
    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device
    )
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    all_y_true = []
    all_y_pred = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for x_seq, y_val in test_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            y_pred = model(x_seq)
            loss = criterion(y_pred.squeeze(), y_val)
            test_loss += loss.item()
            
            all_y_true.extend(y_val.cpu().numpy())
            all_y_pred.extend(y_pred.squeeze().cpu().numpy())
    
    # Calculate final errors
    errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return model, history, model_analysis, errors

def save_and_display_results(model, history, analysis, models_dir, prefix="mixed_lstm_gru_model"):
    """
    Save the model, training plots, and prepare analysis results.
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    
    # Save model
    model_save_path = f"{models_dir}/{prefix}_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    
    # Plot training history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['train_losses'], label="Train Loss")
    ax.plot(history['val_losses'], label="Validation Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True)
    
    plot_save_path = f"{models_dir}/training_plot_{timestamp}.png"
    fig.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close plot to avoid displaying it automatically in Streamlit
    
    return model_save_path, plot_save_path

def load_model(model_path, input_size, hidden_size_lstm=128, hidden_size_gru=128, num_layers=2):
    """
    Load a saved model from disk.
    """
    model = ImprovedMixedRNNModel(
        input_size=input_size,
        hidden_size_lstm=hidden_size_lstm,
        hidden_size_gru=hidden_size_gru,
        num_layers=num_layers
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def run_existing_model(model_path, data_path, ticker):
    """
    Load and run predictions with an existing model, returning the errors dictionary.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize dataset to get input size
    dataset = StockOptionDataset(csv_file=data_path, ticker=ticker)
    
    # Load model
    model = load_model(model_path, input_size=dataset.n_features)
    model = model.to(device)
    model.eval()
    
    # Create data loader for the entire dataset
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for x_seq, y_val in data_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            y_pred = model(x_seq)
            all_y_true.extend(y_val.cpu().numpy())
            all_y_pred.extend(y_pred.squeeze().cpu().numpy())
    
    errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
    return errors


# -----------------------------
# FRONTEND CODE (Streamlit App)
# -----------------------------

def main_app():
    st.title("Option Trading Model - Streamlit UI")
    st.write("A demo application to train and evaluate an LSTM+GRU model for option pricing.")

    # Sidebar configuration / menu
    menu = ["Home", "Train New Model", "Run Existing Model", "Analyze Architecture"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Common config placeholders
    # Adjust these paths as needed (or make them user-configurable)
    default_data_path = "/Users/bekheet/dev/option-ml-prediction/data_files/option_data_scaled.csv"
    default_models_dir = "/Users/bekheet/dev/option-ml-prediction/models"
    
    if choice == "Home":
        st.subheader("Home")
        st.write("Use the sidebar to navigate the available features.")

    elif choice == "Train New Model":
        st.subheader("Train a New Model")
        
        # Let user select CSV path and model directory if desired
        data_path = st.text_input("Data CSV Path", value=default_data_path)
        models_dir = st.text_input("Models Directory", value=default_models_dir)

        # Show available tickers (only if file is readable)
        try:
            tickers, counts = get_available_tickers(data_path)
            if tickers:
                selected_ticker = st.selectbox(
                    "Select Ticker to Train On",
                    options=tickers
                )
                count_index = tickers.index(selected_ticker)
                st.write(f"Data points for {selected_ticker}: {counts[count_index]:,}")
            else:
                st.error("No tickers found. Check your CSV file.")
                return
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return
        
        # Training parameters
        seq_len = st.number_input("Sequence Length", min_value=1, max_value=100, value=15)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=2048, value=128)
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=20)
        hidden_size_lstm = st.number_input("Hidden Size (LSTM)", min_value=1, max_value=2048, value=128)
        hidden_size_gru = st.number_input("Hidden Size (GRU)", min_value=1, max_value=2048, value=128)
        num_layers = st.number_input("Number of Layers (LSTM/GRU)", min_value=1, max_value=10, value=2)
        
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                try:
                    model, history, analysis, final_test_errors = train_option_model(
                        data_path=data_path,
                        ticker=selected_ticker,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        epochs=epochs,
                        hidden_size_lstm=hidden_size_lstm,
                        hidden_size_gru=hidden_size_gru,
                        num_layers=num_layers
                    )
                    
                    # Save results
                    model_save_path, plot_save_path = save_and_display_results(
                        model, history, analysis, models_dir=models_dir
                    )
                    
                    # Display results
                    st.success(f"Model trained and saved to: {model_save_path}")
                    st.image(plot_save_path, caption="Training Loss Curve")
                    
                    st.write("**Final Test Set Metrics**")
                    st.json(final_test_errors)
                    
                    st.write("**Model Architecture Analysis**")
                    st.write(f"Total parameters: {analysis['total_parameters']:,}")
                    st.write(f"Trainable parameters: {analysis['trainable_parameters']:,}")
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")

    elif choice == "Run Existing Model":
        st.subheader("Run Predictions with Existing Model")

        model_dir = st.text_input("Models Directory", value=default_models_dir)
        data_path = st.text_input("Data CSV Path", value=default_data_path)

        # List possible .pth files
        try:
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        except FileNotFoundError:
            st.error("Model directory not found. Please check the path.")
            return

        if not model_files:
            st.warning("No .pth files found in the given directory.")
            return
        
        selected_model_file = st.selectbox("Select a saved model file", model_files)
        
        # Show available tickers (only if file is readable)
        try:
            tickers, counts = get_available_tickers(data_path)
            if tickers:
                selected_ticker = st.selectbox(
                    "Select Ticker to Predict",
                    options=tickers
                )
                count_index = tickers.index(selected_ticker)
                st.write(f"Data points for {selected_ticker}: {counts[count_index]:,}")
            else:
                st.error("No tickers found. Check your CSV file.")
                return
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return
        
        if st.button("Run Predictions"):
            model_path = os.path.join(model_dir, selected_model_file)
            with st.spinner("Running predictions..."):
                try:
                    errors = run_existing_model(model_path, data_path, selected_ticker)
                    st.write("**Prediction Metrics**")
                    st.json(errors)
                except Exception as e:
                    st.error(f"Error running predictions: {e}")

    elif choice == "Analyze Architecture":
        st.subheader("Analyze Default Network Architecture")
        st.write("Create a default model and analyze its layers and parameter counts.")

        # Let user specify some parameters
        input_size = st.number_input("Input Size", 1, 1000, 23)
        seq_len = st.number_input("Sequence Length", 1, 100, 15)
        hidden_size_lstm = st.number_input("Hidden Size (LSTM)", 1, 2048, 128)
        hidden_size_gru = st.number_input("Hidden Size (GRU)", 1, 2048, 128)
        num_layers = st.number_input("Number of Layers", 1, 10, 2)
        batch_size = st.number_input("Batch Size (for shape analysis)", 1, 512, 32)
        
        if st.button("Analyze Model"):
            with st.spinner("Analyzing..."):
                model = ImprovedMixedRNNModel(
                    input_size=input_size,
                    hidden_size_lstm=hidden_size_lstm,
                    hidden_size_gru=hidden_size_gru,
                    num_layers=num_layers
                )
                analysis = analyze_model_architecture(
                    model=model,
                    input_size=input_size,
                    seq_len=seq_len,
                    batch_size=batch_size
                )
                st.write(f"**Total parameters**: {analysis['total_parameters']:,}")
                st.write(f"**Trainable parameters**: {analysis['trainable_parameters']:,}")
                
                st.write("**Layer Shapes**:")
                for layer_name, shapes in analysis['layer_shapes'].items():
                    st.write(f"- **{layer_name}**")
                    st.write(f"  - Input shape: {shapes['input_shape']}")
                    st.write(f"  - Output shape: {shapes['output_shape']}")


def main():
    # Run the Streamlit application
    main_app()

if __name__ == "__main__":
    main()
