import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Stock Option Dataset Handler
# This class prepares financial data for deep learning by:
# - Loading and cleaning option chain data from CSV
# - Creating sequential data windows for time series prediction
# - Handling feature selection and target variable preparation
class StockOptionDataset(Dataset):
    def __init__(self, csv_file, ticker="CGC", seq_len=10):
        super().__init__()
        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Focus on a single stock ticker for consistency
        df = df[df['ticker'] == ticker].copy()
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # Features used for prediction, including:
        # - Option-specific metrics (strike, bid/ask, implied volatility)
        # - Stock metrics (volume, price data)
        # - Technical indicators (moving averages)
        # - Time-based features (day of week/month/year)
        feature_cols = [
            "strike", "bid", "ask", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume",
            "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d",
            "day_of_week", "day_of_month", "day_of_year"
        ]
        
        # We're predicting the option's last trading price
        target_col = "lastPrice"
        
        # Remove any rows with missing data to ensure clean training
        df.dropna(subset=feature_cols + [target_col], inplace=True)

        # Convert dataframe to numpy for faster processing
        data_np = df[feature_cols + [target_col]].to_numpy(dtype=np.float32)
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_features = len(feature_cols)
        self.seq_len = seq_len
        self.data = data_np
        self.n_samples = self.data.shape[0]
        
        # Calculate the maximum valid starting index for sequences
        # We need enough data points ahead of each start point to form a complete sequence
        self.max_index = self.n_samples - self.seq_len - 1
        
    def __len__(self):
        return max(0, self.max_index)
    
    def __getitem__(self, idx):
        # Create a window of historical data (sequence)
        x_seq = self.data[idx : idx + self.seq_len, :self.n_features]
        
        # Get the next price point as our prediction target
        y_val = self.data[idx + self.seq_len, self.n_features]
        
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)


# Hybrid Neural Network Architecture
# Combines LSTM and GRU layers for robust time series processing:
# - LSTM captures long-term dependencies
# - GRU provides efficient processing of recent information
# - Final dense layer generates the price prediction
class MixedRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size_lstm=32, hidden_size_gru=32, num_layers_lstm=1, num_layers_gru=1):
        super(MixedRNNModel, self).__init__()
        
        # LSTM processes the initial sequence, good at capturing long-term patterns
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers_lstm,
            batch_first=True
        )
        
        # GRU refines the LSTM output, efficient at handling recent information
        self.gru = nn.GRU(
            input_size=hidden_size_lstm,
            hidden_size=hidden_size_gru,
            num_layers=num_layers_gru,
            batch_first=True
        )
        
        # Final layer converts processed sequence into a single price prediction
        self.fc = nn.Linear(hidden_size_gru, 1)

    def forward(self, x):
        # Process steps:
        # 1. LSTM extracts temporal patterns from input sequence
        lstm_out, (_, _) = self.lstm(x)
        
        # 2. GRU further processes these patterns
        gru_out, _ = self.gru(lstm_out)
        
        # 3. Take the final time step's output as our sequence summary
        final_hidden = gru_out[:, -1, :]
        
        # 4. Generate final price prediction
        return self.fc(final_hidden)


# Model Training Pipeline
# Handles the complete training process including:
# - Forward/backward passes
# - Loss calculation
# - Validation assessment
# - Progress tracking
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=5,
    lr=1e-3,
    device='cpu'
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        for x_seq, y_val in train_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            
            # Standard training step: predict, calculate loss, update weights
            optimizer.zero_grad()
            y_pred = model(x_seq)
            loss = criterion(y_pred.squeeze(), y_val)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_seq, y_val in val_loader:
                x_seq, y_val = x_seq.to(device), y_val.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred.squeeze(), y_val)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses


# Main Execution Pipeline
# Orchestrates the entire model training process:
# 1. Data preparation
# 2. Model initialization
# 3. Training execution
# 4. Performance visualization
# 5. Model saving
def main():
    # Configuration
    data_path = "/Users/bekheet/dev/option-ml-prediction/data_files/option_data_scaled.csv"
    ticker = "CGC"
    seq_len = 10
    batch_size = 64
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize and split dataset
    dataset = StockOptionDataset(csv_file=data_path, ticker=ticker, seq_len=seq_len)
    
    if len(dataset) < 1:
        raise ValueError("Insufficient data for sequence creation!")
    
    # Split data: 70% training, 15% validation, 15% testing
    total_len = len(dataset)
    train_len = int(0.70 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset splits: {len(train_ds)} train, {len(val_ds)} validation, {len(test_ds)} test samples")
    
    # Create data loaders for batch processing
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize and train model
    model = MixedRNNModel(input_size=dataset.n_features, hidden_size_lstm=32, hidden_size_gru=32)
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device
    )
    
    # Save trained model
    model_save_path = "mixed_lstm_gru_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Visualize training progress
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.title(f"Training and Validation Loss Curves for {ticker}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Final model evaluation on test set
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x_seq, y_val in test_loader:
            x_seq, y_val = x_seq.to(device), y_val.to(device)
            y_pred = model(x_seq)
            loss = criterion(y_pred.squeeze(), y_val)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Final Test Set MSE: {test_loss:.6f}")
    

if __name__ == "__main__":
    main()