"""
Training Utilities for TRAF-GNN
Helper functions for reproducibility, device management, and data handling
"""

import torch
import numpy as np
import random
import json
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Set random seed to {seed}")


def get_device(device_str: str = 'auto') -> torch.device:
    """
    Get torch device
    
    Args:
        device_str: 'auto', 'cuda', or 'cpu'
    
    Returns:
        torch.device
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    if device.type == 'cuda':
        print(f"✓ Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"✓ Using device: {device}")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, input_shape: tuple = None):
    """
    Print model architecture summary
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for model
    """
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Print layer-wise parameters
    print("\nLayer-wise parameters:")
    print("-" * 70)
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total += num_params
            print(f"  {name:40s} {num_params:>10,}")
    
    print("-" * 70)
    print(f"  {'Total':40s} {total:>10,}")
    print("=" * 70 + "\n")


def load_scaler(stats_path: str) -> dict:
    """
    Load normalization statistics
    
    Args:
        stats_path: Path to stats JSON file
    
    Returns:
        Dictionary with 'mean' and 'std' as numpy arrays
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Convert lists to numpy arrays
    scaler = {
        'mean': np.array(stats['mean']),
        'std': np.array(stats['std'])
    }
    
    return scaler


def denormalize(data: np.ndarray, scaler: dict) -> np.ndarray:
    """
    Denormalize data using scaler
    
    Args:
        data: Normalized data
        scaler: Dictionary with 'mean' and 'std'
    
    Returns:
        Denormalized data
    """
    mean = scaler['mean']
    std = scaler['std']
    
    # Handle different shapes
    if data.ndim == 4:  # (batch, pred_horizon, num_nodes, features)
        mean = mean.reshape(1, 1, -1, 1)
        std = std.reshape(1, 1, -1, 1)
    elif data.ndim == 3:  # (batch, num_nodes, features)
        mean = mean.reshape(1, -1, 1)
        std = std.reshape(1, -1, 1)
    elif data.ndim == 2:  # (timesteps, num_nodes)
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
    
    return data * std + mean


def save_predictions(predictions: np.ndarray, targets: np.ndarray, 
                     save_path: str, sample_indices: list = None):
    """
    Save predictions and targets for analysis
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        save_path: Path to save numpy file
        sample_indices: Optional indices to save specific samples
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if sample_indices is not None:
        predictions = predictions[sample_indices]
        targets = targets[sample_indices]
    
    np.savez(
        save_path,
        predictions=predictions,
        targets=targets
    )
    
    print(f"✓ Saved predictions to {save_path}")


def create_experiment_dir(experiment_name: str, base_dir: str = 'experiments') -> Path:
    """
    Create experiment directory with timestamp
    
    Args:
        experiment_name: Name of experiment
        base_dir: Base directory for experiments
    
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    
    print(f"✓ Created experiment directory: {exp_dir}")
    
    return exp_dir


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0, 
                 min_epochs: int = 10):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            min_epochs: Minimum number of epochs before early stopping can trigger
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
        
        Returns:
            True if should stop, False otherwise
        """
        self.epoch = epoch
        
        # Don't stop before minimum epochs
        if epoch < self.min_epochs:
            if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            return False
        
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                print(f"   Best validation loss: {self.best_loss:.4f}")
                return True
        
        return False


if __name__ == '__main__':
    print("Testing utilities module...")
    print("=" * 70)
    
    # Test set_seed
    set_seed(42)
    
    # Test get_device
    device = get_device('auto')
    
    # Test early stopping
    print("\nTesting early stopping...")
    early_stop = EarlyStopping(patience=3, min_epochs=2)
    
    losses = [5.0, 4.5, 4.0, 4.1, 4.2, 4.3, 4.4]
    for epoch, loss in enumerate(losses):
        should_stop = early_stop(loss, epoch)
        print(f"  Epoch {epoch}: loss={loss:.2f}, counter={early_stop.counter}, stop={should_stop}")
        if should_stop:
            break
    
    print("\n✓ Utilities module test passed!")
