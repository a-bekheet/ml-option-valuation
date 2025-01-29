import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
import os

def get_available_tickers(csv_file):
    """Load and return sorted list of unique tickers with their data counts."""
    df = pd.read_csv(csv_file)
    ticker_counts = df['ticker'].value_counts().sort_index()
    return list(ticker_counts.index), list(ticker_counts.values)

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
    def __init__(self, csv_file, ticker="CGC", seq_len=15):
        super().__init__()
        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        
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
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Custom learning rate logging
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
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses

def main():
    # Configuration
    data_path = "/Users/bekheet/dev/option-ml-prediction/data_files/option_data_scaled.csv"
    
    # Get available tickers and their counts, let user select one
    tickers, counts = get_available_tickers(data_path)
    ticker = select_ticker(tickers, counts)
    
    seq_len = 15
    batch_size = 128
    epochs = 20
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
    
    # Initialize improved model
    model = ImprovedMixedRNNModel(
        input_size=dataset.n_features,
        hidden_size_lstm=128,
        hidden_size_gru=128,
        num_layers=2
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
    
    # Create models directory if it doesn't exist
    models_dir = "/Users/bekheet/dev/option-ml-prediction/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save trained model with timestamp
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    model_save_path = f"{models_dir}/mixed_lstm_gru_model_{ticker}_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot and save results
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title(f"Training and Validation Loss Curves for {ticker}")
    plt.suptitle(f"Model Training Results - {timestamp}", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    # Save plot with matching filename pattern
    plot_save_path = f"{models_dir}/training_plot_{ticker}_{timestamp}.png"
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_save_path}")
    
    plt.show()
    
    # Final evaluation
    model.eval()
    test_loss = 0.0
    all_y_true = []
    all_y_pred = []
    criterion = nn.MSELoss()
    
    print("\nPerforming final model evaluation...")
    print("-" * 50)
    
    with torch.no_grad():
        for x_seq, y_val in test_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            y_pred = model(x_seq)
            loss = criterion(y_pred.squeeze(), y_val)
            test_loss += loss.item()
            
            # Collect predictions and actual values
            all_y_true.extend(y_val.cpu().numpy())
            all_y_pred.extend(y_pred.squeeze().cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Convert to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Calculate comprehensive error metrics
    errors = calculate_errors(torch.tensor(all_y_true), torch.tensor(all_y_pred))
    
    # Print detailed error analysis
    print("\nFinal Test Set Metrics:")
    print("-" * 50)
    print(f"Mean Squared Error (MSE): {errors['mse']:.6f}")
    print(f"Root Mean Squared Error (RMSE): {errors['rmse']:.6f}")
    print(f"Mean Absolute Error (MAE): {errors['mae']:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {errors['mape']:.2f}%")
    print("-" * 50)
    
    # Additional statistics
    print("\nPrediction Statistics:")
    print("-" * 50)
    print(f"Average True Value: {np.mean(all_y_true):.6f}")
    print(f"Average Predicted Value: {np.mean(all_y_pred):.6f}")
    print(f"Standard Deviation of Error: {np.std(all_y_true - all_y_pred):.6f}")
    print(f"Maximum Absolute Error: {np.max(np.abs(all_y_true - all_y_pred)):.6f}")
    print(f"Minimum Absolute Error: {np.min(np.abs(all_y_true - all_y_pred)):.6f}")

if __name__ == "__main__":
    main()