import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

class StockOptionDataset(Dataset):
    def __init__(self, csv_file, ticker="CGC", seq_len=15):  # Vary sequence length here
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
        
        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Multi-layer LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Batch normalization between LSTM and GRU
        self.mid_bn = nn.BatchNorm1d(hidden_size_lstm)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        
        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=hidden_size_lstm,
            hidden_size=hidden_size_gru,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Final layers
        self.bn_final = nn.BatchNorm1d(hidden_size_gru)
        self.fc1 = nn.Linear(hidden_size_gru, hidden_size_gru // 2)
        self.fc2 = nn.Linear(hidden_size_gru // 2, 1)
        
    def forward(self, x):
        # Input normalization
        batch_size, seq_len, features = x.size()
        x = x.view(-1, features)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Mid-processing
        lstm_out = lstm_out.contiguous()
        batch_size, seq_len, hidden_size = lstm_out.size()
        lstm_out = lstm_out.view(-1, hidden_size)
        lstm_out = self.mid_bn(lstm_out)
        lstm_out = lstm_out.view(batch_size, seq_len, hidden_size)
        lstm_out = self.dropout(lstm_out)
        
        # GRU processing
        gru_out, _ = self.gru(lstm_out)
        
        # Take final time step
        final_out = gru_out[:, -1, :]
        
        # Final processing
        final_out = self.bn_final(final_out)
        final_out = self.dropout(final_out)
        final_out = torch.relu(self.fc1(final_out))
        final_out = self.fc2(final_out)
        
        return final_out


# Early stopping implementation
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


# Improved training function with learning rate scheduling and gradient clipping
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,  # Increased epochs since we have early stopping
    lr=1e-3,
    device='cpu'
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
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
            
            # Gradient clipping
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
        
        # Store losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        early_stopping(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses


def main():
    # Configuration
    data_path = "/Users/bekheet/dev/option-ml-prediction/data_files/option_data_scaled.csv"
    ticker = "CGC"
    seq_len = 15  # Increased from 10
    batch_size = 128  # Increased from 64
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
    
    # Save trained model with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    model_save_path = f"/Users/bekheet/dev/option-ml-prediction/models/mixed_lstm_gru_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title(f"Training and Validation Loss Curves for {ticker}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Final evaluation
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
    print(f"Final Test Set RMSE: {np.sqrt(test_loss):.6f}")
    print(f"Final Test Set Percentage Error: {np.sqrt(test_loss) * 100:.2f}%")


if __name__ == "__main__":
    main()