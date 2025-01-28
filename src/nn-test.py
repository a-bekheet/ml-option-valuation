import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------------------------------------------------------
# 1) DATASET CLASS
# --------------------------------------------------------------------------
class OptionDataset(Dataset):
    """
    A PyTorch Dataset that:
      - Loads a scaled CSV file into memory (for this simple example).
      - Creates small sequences of features to feed into an RNN.
      - Uses 'lastPrice' as the target (just an example).
    """

    def __init__(self, csv_file, sequence_length=5):
        super().__init__()
        # Read the scaled CSV
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        
        # For this example, we'll pick a subset of columns as features.
        # Exclude 'lastPrice' from the input features since we'll use it as the target.
        feature_cols = [
            "strike", "bid", "ask", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "daysToExpiry", "stockVolume",
            "stockClose", "stockAdjClose", "stockOpen", "stockHigh", "stockLow",
            "strikeDelta", "stockClose_ewm_5d", "stockClose_ewm_15d",
            "stockClose_ewm_45d", "stockClose_ewm_135d",
            "day_of_week", "day_of_month", "day_of_year"
            # ... add or remove columns as needed
        ]
        
        # Filter the dataframe to these columns + 'lastPrice'
        self.df = self.df[feature_cols + ["lastPrice"]].copy()
        
        # Drop any rows with NaNs (should be none if your pipeline handled them, but just in case)
        self.df.dropna(inplace=True)
        
        # Convert dataframe to numpy for easier slicing
        self.data = self.df.to_numpy(dtype=np.float32)
        
        self.n_samples = self.data.shape[0]
        self.n_features = len(feature_cols)
        self.sequence_length = sequence_length
        
        # We'll generate input sequences of shape (sequence_length, n_features)
        # and output a single value (lastPrice) from the "next" time step,
        # OR the same time step. Here, we'll keep it simple:
        #   X[i] = data[i : i + sequence_length,  feature_cols]
        #   y[i] = data[i + sequence_length, lastPrice]  (predict next step)
        
        # This means the last "sequence_length" rows can't form a full sequence (if we do next-step),
        # so the dataset length is effectively n_samples - sequence_length.
        self.max_index = self.n_samples - self.sequence_length - 1

    def __len__(self):
        return max(0, self.max_index)

    def __getitem__(self, idx):
        # Get sequence of features from i to i+sequence_length
        x_seq = self.data[idx : idx + self.sequence_length, :self.n_features]
        
        # We'll predict 'lastPrice' 1 step ahead
        # i.e., at i + sequence_length
        y_val = self.data[idx + self.sequence_length, self.n_features]  # lastPrice is the last column
        
        # Convert to torch Tensors
        x_seq = torch.tensor(x_seq, dtype=torch.float32)   # shape: (sequence_length, n_features)
        y_val = torch.tensor(y_val, dtype=torch.float32)   # shape: (1,)
        
        # For RNNs, we typically want the shape: (sequence_length, batch_size, n_features).
        # We'll handle the batch dimension in the DataLoader, so for a single sample:
        # we can either keep (sequence_length, n_features) or (sequence_length, 1, n_features).
        # We'll keep (sequence_length, n_features) and let the DataLoader handle the rest.
        return x_seq, y_val

# --------------------------------------------------------------------------
# 2) DEFINE A SIMPLE RNN MODEL
# --------------------------------------------------------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # We use a simple RNN cell. You could use nn.LSTM or nn.GRU as well.
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        # Final fully connected layer to map hidden state -> output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        # Initialize hidden state for RNN
        # shape: (num_layers, batch_size, hidden_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # RNN forward pass
        out, hn = self.rnn(x, h0)  # out shape: (batch_size, seq_len, hidden_size)
        
        # We take the last timestep's output for prediction
        last_out = out[:, -1, :]   # shape: (batch_size, hidden_size)
        # Map to final output (e.g. predict lastPrice)
        y_pred = self.fc(last_out) # shape: (batch_size, 1)
        return y_pred

# --------------------------------------------------------------------------
# 3) TRAINING / EVALUATION
# --------------------------------------------------------------------------
def train_rnn_model(dataset_path, sequence_length=5, epochs=3, batch_size=64, hidden_size=32):
    # Create dataset
    dataset = OptionDataset(csv_file=dataset_path, sequence_length=sequence_length)
    
    # If there's not enough data to form sequences:
    if len(dataset) == 0:
        raise ValueError("No valid sequences found. Possibly your dataset is too short or has NaNs.")
    
    # Train/Val Split
    # We'll do a simple 80/20 split
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Instantiate model
    model = SimpleRNNModel(input_size=dataset.n_features, hidden_size=hidden_size, num_layers=1)
    model = model.to("cpu")  # or "cuda" if you have a GPU
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # ----------------------
        # TRAINING LOOP
        # ----------------------
        model.train()
        total_loss = 0.0
        for batch_idx, (x_seq, y_val) in enumerate(train_loader):
            # x_seq shape: (batch_size, seq_len, n_features)
            # y_val shape: (batch_size,)
            optimizer.zero_grad()
            
            # Forward
            y_pred = model(x_seq)
            # y_pred shape: (batch_size, 1)
            
            loss = criterion(y_pred.squeeze(), y_val)  # shape check
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # ----------------------
        # VALIDATION LOOP
        # ----------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, y_val in val_loader:
                y_pred = model(x_seq)
                loss = criterion(y_pred.squeeze(), y_val)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print("Training complete!")
    return model

# --------------------------------------------------------------------------
# 4) MAIN / USAGE EXAMPLE
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Assume your scaled file is here:
    scaled_file = "/Users/bekheet/dev/option-ml-prediction/data_files/option_data_scaled.csv"
    
    # Run a small training test
    model = train_rnn_model(
        dataset_path=scaled_file,
        sequence_length=5,
        epochs=3,
        batch_size=64,
        hidden_size=32
    )
    
    # You could save the model for later use:
    # torch.save(model.state_dict(), "simple_rnn_test_model.pt")
