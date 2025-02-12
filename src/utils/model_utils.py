import torch
import torch.nn as nn
import os
from datetime import datetime
from torch.utils.data import DataLoader, Subset
import numpy as np

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

def load_model(model_path, model_class, input_size, hidden_size_lstm=128, hidden_size_gru=128, num_layers=2, output_size=1):
    model = model_class(
        input_size=input_size,
        hidden_size_lstm=hidden_size_lstm,
        hidden_size_gru=hidden_size_gru,
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