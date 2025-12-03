"""
Evaluation Metrics for Traffic Forecasting
Implements MAE, RMSE, and MAPE with masking support
"""

import torch
import numpy as np


def masked_mae(preds, labels, null_val=0.0):
    """
    Masked Mean Absolute Error
    
    Args:
        preds: Predictions tensor (batch, pred_horizon, num_nodes, features)
        labels: Ground truth tensor (same shape as preds)
        null_val: Value to mask (default: 0.0)
    
    Returns:
        MAE value (float)
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    """
    Masked Root Mean Squared Error
    
    Args:
        preds: Predictions tensor
        labels: Ground truth tensor
        null_val: Value to mask
    
    Returns:
        RMSE value (float)
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.sqrt(torch.mean(loss))


def masked_mape(preds, labels, null_val=0.0, min_threshold=5.0):
    """
    Masked Mean Absolute Percentage Error
    
    Args:
        preds: Predictions tensor
        labels: Ground truth tensor
        null_val: Value to mask
        min_threshold: Minimum speed threshold to avoid division by very small values
                      (default: 5.0 mph - only calculate MAPE for speeds > 5 mph)
    
    Returns:
        MAPE value (percentage)
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    # Additional mask: only consider values above threshold to avoid division issues
    threshold_mask = torch.abs(labels) > min_threshold
    mask = mask & threshold_mask
    
    mask = mask.float()
    
    # If no valid values after masking, return 0
    if torch.sum(mask) == 0:
        return torch.tensor(0.0)
    
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # Calculate percentage error only for non-zero labels
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss) * 100  # Return as percentage


def calculate_metrics(preds, labels, scaler=None, null_val=0.0):
    """
    Calculate all metrics (MAE, RMSE, MAPE)
    
    Args:
        preds: Predictions tensor or numpy array
        labels: Ground truth tensor or numpy array
        scaler: Dictionary with 'mean' and 'std' for denormalization (optional)
        null_val: Value to mask
    
    Returns:
        Dictionary with 'MAE', 'RMSE', 'MAPE' keys
    """
    # Convert to torch tensors if numpy
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).float()
    
    # Denormalize if scaler provided
    if scaler is not None:
        mean = torch.tensor(scaler['mean']).float()
        std = torch.tensor(scaler['std']).float()
        
        # Handle different shapes
        if preds.dim() == 4:  # (batch, pred_horizon, num_nodes, features)
            mean = mean.view(1, 1, -1, 1)
            std = std.view(1, 1, -1, 1)
        elif preds.dim() == 3:  # (batch, num_nodes, features)
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)
        
        preds = preds * std + mean
        labels = labels * std + mean
    
    # Calculate metrics
    mae = masked_mae(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


class MetricTracker:
    """Track metrics over multiple batches"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.mae_sum = 0.0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0
        self.count = 0
    
    def update(self, preds, labels, scaler=None, null_val=0.0):
        """Update metrics with new batch"""
        metrics = calculate_metrics(preds, labels, scaler, null_val)
        
        self.mae_sum += metrics['MAE']
        self.rmse_sum += metrics['RMSE']
        self.mape_sum += metrics['MAPE']
        self.count += 1
    
    def get_metrics(self):
        """Get average metrics"""
        if self.count == 0:
            return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0}
        
        return {
            'MAE': self.mae_sum / self.count,
            'RMSE': self.rmse_sum / self.count,
            'MAPE': self.mape_sum / self.count
        }


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics module...")
    
    # Create dummy data
    preds = torch.randn(32, 3, 207, 1) * 10 + 50
    labels = torch.randn(32, 3, 207, 1) * 10 + 50
    
    # Test individual metrics
    mae = masked_mae(preds, labels)
    rmse = masked_rmse(preds, labels)
    mape = masked_mape(preds, labels)
    
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Test calculate_metrics
    metrics = calculate_metrics(preds, labels)
    print(f"\nAll metrics: {metrics}")
    
    # Test MetricTracker
    tracker = MetricTracker()
    tracker.update(preds, labels)
    tracker.update(preds, labels)
    avg_metrics = tracker.get_metrics()
    print(f"\nAverage metrics: {avg_metrics}")
    
    print("\n✓ Metrics module test passed!")
