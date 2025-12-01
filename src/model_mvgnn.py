"""
TRAF-GNN: Multi-View Graph Neural Network for Traffic Forecasting
Complete model architecture integrating all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiViewGCN, AttentionFusion, TemporalModule


class TRAFGNN(nn.Module):
    """
    TRAF-GNN Model Architecture
    
    Multi-view graph learning with efficient neighbor selection for traffic forecasting
    
    Args:
        num_nodes: Number of traffic sensors/nodes
        in_features: Input feature dimension (usually 1 for speed)
        hidden_dim: Hidden dimension for GNN layers
        num_gnn_layers: Number of stacked GNN layers
        num_temporal_layers: Number of GRU layers
        pred_horizon: Number of future timesteps to predict
        dropout: Dropout rate
    """
    
    def __init__(self, num_nodes, in_features=1, hidden_dim=64,
                 num_gnn_layers=2, num_temporal_layers=2,
                 pred_horizon=3, dropout=0.3):
        super(TRAFGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.pred_horizon = pred_horizon
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # Stacked Multi-View GCN layers
        self.gnn_layers = nn.ModuleList([
            MultiViewGCN(
                in_features=hidden_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                num_views=3,
                dropout=dropout
            )
            for i in range(num_gnn_layers)
        ])
        
        # Attention fusion for each GNN layer
        self.fusion_layers = nn.ModuleList([
            AttentionFusion(hidden_dim, num_views=3)
            for _ in range(num_gnn_layers)
        ])
        
        # Temporal module
        self.temporal= TemporalModule(
            hidden_dim=hidden_dim,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_horizon * in_features)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, graphs):
        """
        Forward pass
        
        Args:
            x: Input sequences (batch, seq_length, num_nodes, in_features)
            graphs: Dictionary of adjacency matrices
                   Keys: 'physical', 'proximity', 'correlation'
        
        Returns:
            predictions: (batch, pred_horizon, num_nodes, in_features)
        """
        batch_size, seq_length, num_nodes, in_features = x.size()
        
        # Process each timestep through spatial GNN
        spatial_outputs = []
        
        for t in range(seq_length):
            x_t = x[:, t, :, :]  # (batch, num_nodes, in_features)
            
            # Input projection
            h = self.input_proj(x_t)  # (batch, num_nodes, hidden_dim)
            
            # Stack GNN layers with residual connections
            for gnn_layer, fusion_layer in zip(self.gnn_layers, self.fusion_layers):
                # Multi-view graph convolution
                view_outputs = gnn_layer(h, graphs)
                
                # Fuse views
                h_fused = fusion_layer(view_outputs)
                
                # Residual connection
                h = h + h_fused
            
            spatial_outputs.append(h)
        
        # Stack spatial outputs: (batch, seq_length, num_nodes, hidden_dim)
        spatial_features = torch.stack(spatial_outputs, dim=1)
        
        # Temporal modeling
        temporal_features = self.temporal(spatial_features)  # (batch, num_nodes, hidden_dim)
        
        # Output projection
        output = self.output_proj(temporal_features)  # (batch, num_nodes, pred_horizon * in_features)
        
        # Reshape to (batch, num_nodes, pred_horizon, in_features)
        output = output.view(batch_size, num_nodes, self.pred_horizon, in_features)
        
        # Permute to (batch, pred_horizon, num_nodes, in_features)
        output = output.permute(0, 2, 1, 3)
        
        return output
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_nodes, config=None):
    """
    Create TRAF-GNN model with given configuration
    
    Args:
        num_nodes: Number of traffic sensors
        config: Optional configuration dictionary
    
    Returns:
        model: TRAF-GNN model
    """
    if config is None:
        config = {
            'in_features': 1,
            'hidden_dim': 64,
            'num_gnn_layers': 2,
            'num_temporal_layers': 2,
            'pred_horizon': 3,
            'dropout': 0.3
        }
    
    model = TRAFGNN(
        num_nodes=num_nodes,
        **config
    )
    
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("🚦 Testing TRAF-GNN Model")
    print("=" * 60)
    
    # Configuration
    batch_size = 8
    seq_length = 12
    num_nodes = 207
    in_features = 1
    pred_horizon = 3
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_model(
        num_nodes=num_nodes,
        config={
            'in_features': in_features,
            'hidden_dim': 64,
            'num_gnn_layers': 2,
            'num_temporal_layers': 2,
            'pred_horizon': pred_horizon,
            'dropout': 0.3
        }
    )
    
    print(f"✓ Model created")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Create dummy data
    print("\n📊 Creating test data...")
    x = torch.randn(batch_size, seq_length, num_nodes, in_features)
    
    # Create dummy graphs
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj > 0.95).float()
    adj = adj + adj.T
    
    graphs = {
        'physical': adj,
        'proximity': adj,
        'correlation': adj
    }
    
    print(f"✓ Test data created")
    print(f"  Input shape: {x.shape}")
    print(f"  Graph shapes: {list(graphs.values())[0].shape}")
    
    # Forward pass
    print("\n🧪 Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(x, graphs)
    
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (batch={batch_size}, pred_horizon={pred_horizon}, num_nodes={num_nodes}, features={in_features})")
    
    # Verify shapes
    assert output.shape == (batch_size, pred_horizon, num_nodes, in_features), "Output shape mismatch!"
    
    print("\n" + "=" * 60)
    print("✅ TRAF-GNN Model Test Passed!")
    print("=" * 60)
    
    print("\n📋 Model Summary:")
    print(f"  Architecture: Multi-View GNN + GRU")
    print(f"  Input: (batch, {seq_length}, {num_nodes}, {in_features})")
    print(f"  Output: (batch, {pred_horizon}, {num_nodes}, {in_features})")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Graph views: 3 (physical, proximity, correlation)")
    print(f"  GNN layers: 2")
    print(f"  Temporal layers: 2 (GRU)")
