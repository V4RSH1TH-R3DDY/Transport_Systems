"""
Configuration Management for TRAF-GNN
Centralized hyperparameter and experiment configuration
"""

import json
from pathlib import Path
from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for TRAF-GNN
    
    Returns:
        Dictionary with all configuration parameters
    """
    config = {
        # Model Architecture
        'model': {
            'hidden_dim': 64,
            'num_gnn_layers': 2,
            'num_temporal_layers': 2,
            'dropout': 0.3,
            'in_features': 1,
        },
        
        # Data
        'data': {
            'dataset': 'metr-la',
            'seq_length': 12,  # 1 hour (5-min intervals)
            'pred_horizon': 3,  # 15 minutes
            'batch_size': 32,
            'num_workers': 0,
            'use_demo_graphs': True,
        },
        
        # Training
        'training': {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'grad_clip': 5.0,
            'patience': 15,  # Early stopping patience
            'min_epochs': 10,  # Minimum epochs before early stopping
        },
        
        # Learning Rate Scheduling
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'patience': 5,
            'factor': 0.5,
            'min_lr': 1e-6,
        },
        
        # Paths
        'paths': {
            'data_dir': 'data/processed',
            'graph_dir': 'graphs',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
        },
        
        # Experiment
        'experiment': {
            'name': 'traf_gnn_baseline',
            'seed': 42,
            'device': 'auto',  # 'cuda', 'cpu', or 'auto'
        },
    }
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['model', 'data', 'training', 'paths', 'experiment']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate model config
    model_config = config['model']
    assert model_config['hidden_dim'] > 0, "hidden_dim must be positive"
    assert model_config['num_gnn_layers'] > 0, "num_gnn_layers must be positive"
    assert 0 <= model_config['dropout'] < 1, "dropout must be in [0, 1)"
    
    # Validate data config
    data_config = config['data']
    assert data_config['batch_size'] > 0, "batch_size must be positive"
    assert data_config['seq_length'] > 0, "seq_length must be positive"
    assert data_config['pred_horizon'] > 0, "pred_horizon must be positive"
    
    # Validate training config
    train_config = config['training']
    assert train_config['num_epochs'] > 0, "num_epochs must be positive"
    assert train_config['learning_rate'] > 0, "learning_rate must be positive"
    
    return True


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Saved config to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        filepath: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    # Validate loaded config
    validate_config(config)
    
    print(f"✓ Loaded config from {filepath}")
    return config


def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values
    
    Args:
        base_config: Base configuration
        updates: Dictionary of updates (can be nested)
    
    Returns:
        Updated configuration
    """
    config = base_config.copy()
    
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            # Recursively update nested dictionaries
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    
    return config


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


if __name__ == '__main__':
    # Test configuration system
    print("Testing configuration module...")
    print("=" * 60)
    
    # Get default config
    config = get_default_config()
    
    # Print config
    print("\nDefault Configuration:")
    print("-" * 60)
    print_config(config)
    
    # Validate config
    print("\n" + "=" * 60)
    print("Validating configuration...")
    validate_config(config)
    print("✓ Configuration valid")
    
    # Save config
    print("\n" + "=" * 60)
    save_config(config, 'test_config.json')
    
    # Load config
    loaded_config = load_config('test_config.json')
    
    # Test update
    updates = {
        'model': {'hidden_dim': 128},
        'training': {'learning_rate': 0.0001}
    }
    updated_config = update_config(config, updates)
    print(f"\nUpdated hidden_dim: {updated_config['model']['hidden_dim']}")
    print(f"Updated learning_rate: {updated_config['training']['learning_rate']}")
    
    # Cleanup
    Path('test_config.json').unlink()
    
    print("\n✓ Configuration module test passed!")
