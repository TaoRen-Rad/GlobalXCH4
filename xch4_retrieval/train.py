"""
Training Module
Functions for training the probabilistic CH4 inversion model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import copy
import os
from typing import Tuple, List, Dict

from .model import ProbabilisticMLP, gaussian_nll_loss


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    num_epochs: int = 200,
    batch_size: int = 4096,
    learning_rate: float = 0.005,
    snapshot_window_size: int = 5,
    random_seed: int = 42
) -> Tuple[ProbabilisticMLP, List[Dict], int, float]:
    # Set random seed
    torch.manual_seed(random_seed)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = ProbabilisticMLP(input_size)
    model.to(device)
    print(f"Initialized probabilistic model with {input_size} input features")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler()
    
    # Snapshot ensemble variables
    model_window = deque(maxlen=snapshot_window_size)
    best_val_loss = float('inf')
    best_epoch = -1
    best_snapshots = []
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            with autocast():
                optimizer.zero_grad()
                mean, variance = model(X_batch)
                loss = gaussian_nll_loss(mean, variance, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                mean, variance = model(X_batch)
                loss = gaussian_nll_loss(mean, variance, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update snapshot window
        current_model_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        model_window.append(current_model_state)
        
        # Update learning rate
        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress for every epoch
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Learning Rate: {current_lr:.6f}")
        
        # Check for best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_snapshots = [copy.deepcopy(state) for state in model_window]
            print(f"*** New best validation loss: {best_val_loss:.6f} (epoch {epoch+1}) ***")
    
    print(f"\nTraining completed! Best model from epoch {best_epoch+1}, "
          f"validation loss: {best_val_loss:.6f}")
    
    return model, best_snapshots, best_epoch, best_val_loss


def save_model(
    snapshots: List[Dict],
    best_epoch: int,
    best_val_loss: float,
    output_mean: float,
    output_scale: float,
    feature_columns: List[str],
    save_path: str
):

    checkpoint = {
        'snapshots': snapshots,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'output_mean': output_mean,
        'output_scale': output_scale,
        'feature_columns': feature_columns
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


def load_model(checkpoint_path: str, device: torch.device) -> Dict:

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Model loaded from: {checkpoint_path}")
    return checkpoint

