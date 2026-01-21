import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import copy
from typing import Tuple, List, Dict

from .model import ProbabilisticMLP, gaussian_nll_loss
from .train import load_model


def finetune_model(
    pretrained_model_path: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    num_epochs: int = 65,
    batch_size: int = 512,
    learning_rate: float = 0.0004,
    snapshot_window_size: int = 5,
    random_seed: int = 42
) -> Tuple[ProbabilisticMLP, List[Dict], int, float]:
    """Fine-tune pre-trained model on GOSAT data."""
    torch.manual_seed(random_seed)
    
    print("Loading pre-trained model...")
    checkpoint = load_model(pretrained_model_path, device)
    pretrained_state = checkpoint['snapshots'][-1]
    saved_feature_columns = checkpoint.get('feature_columns', [])
    
    input_size = X_train.shape[1]
    model = ProbabilisticMLP(input_size)
    
    if len(saved_feature_columns) != input_size:
        print(f"Warning: Pre-trained model has {len(saved_feature_columns)} features, "
              f"but data has {input_size} features")
    
    model.load_state_dict(pretrained_state)
    model.to(device)
    print("Pre-trained model loaded successfully")
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=12, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler()
    
    model_window = deque(maxlen=snapshot_window_size)
    best_val_loss = float('inf')
    best_epoch = -1
    best_snapshots = []
    
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
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
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                mean, variance = model(X_batch)
                loss = gaussian_nll_loss(mean, variance, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        current_model_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        model_window.append(current_model_state)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_snapshots = [copy.deepcopy(state) for state in model_window]
            print(f"*** New best validation loss: {best_val_loss:.6f} (epoch {epoch+1}) ***")
        
        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.6f}")
    
    print(f"\nFine-tuning completed! Best model from epoch {best_epoch+1}, "
          f"validation loss: {best_val_loss:.6f}")
    
    return model, best_snapshots, best_epoch, best_val_loss

