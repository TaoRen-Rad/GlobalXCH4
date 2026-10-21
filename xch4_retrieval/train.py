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
) -> Tuple[ProbabilisticMLP, List[Dict], List[float], List[int]]:
    """Train and retain the last complete cosine-cycle snapshots."""
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
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler()
    
    # Snapshot ensemble variables
    snapshots = deque(maxlen=snapshot_window_size)
    snapshot_epochs = deque(maxlen=snapshot_window_size)
    
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
                mean, log_sigma = model(X_batch)
                loss = gaussian_nll_loss(mean, log_sigma, y_batch)
            
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
                mean, log_sigma = model(X_batch)
                loss = gaussian_nll_loss(mean, log_sigma, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Save at the cycle end, before scheduler.step() restarts the cycle.
        if scheduler.T_cur == scheduler.T_i - 1:
            snapshots.append(copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
            snapshot_epochs.append(epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress for every epoch
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Learning Rate: {current_lr:.6f}")
        
        scheduler.step(epoch + 1)

    if len(snapshots) < snapshot_window_size:
        raise RuntimeError(f"Only {len(snapshots)} complete-cycle snapshots; {snapshot_window_size} required")
    print(f"\nTraining completed! Snapshot epochs: {list(snapshot_epochs)}")
    return model, list(snapshots), val_losses, list(snapshot_epochs)


def save_model(
    snapshots: List[Dict],
    validation_losses: List[float],
    snapshot_epochs: List[int],
    output_mean: float,
    output_scale: float,
    feature_columns: List[str],
    save_path: str
):

    checkpoint = {
        'snapshots': snapshots,
        'validation_losses': validation_losses,
        'snapshot_epochs': snapshot_epochs,
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
