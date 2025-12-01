"""
Graph Convolutional Network Layers for TRAF-GNN
Implements GCN and multi-view fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer: H' = σ(D^{-1/2} A D^{-1/2} H W)
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to add bias
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
        
        Returns:
            output: Transformed features (batch, num_nodes, out_features)
        """
        # Normalize adjacency matrix: D^{-1/2} A D^{-1/2}
        adj_normalized = self.normalize_adj(adj)
        
        # Linear transformation: H W
        support = torch.matmul(x, self.weight)  # (batch, num_nodes, out_features)
        
        # Graph convolution: A_norm H W
        output = torch.matmul(adj_normalized, support)  # (batch, num_nodes, out_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    @staticmethod
    def normalize_adj(adj):
        """
        Symmetric normalization: D^{-1/2} A D^{-1/2}
        """
        # Add self-loops if not present
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Degree matrix
        degree = torch.sum(adj, dim=1)
        
        # D^{-1/2}
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        
        # D^{-1/2} A D^{-1/2}
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        return adj_normalized


class MultiViewGCN(nn.Module):
    """
    Multi-view Graph Convolutional Network
    Processes multiple graph views separately then fuses them
    
    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden dimension  
        num_views: Number of graph views (default: 3)
        dropout: Dropout rate
    """
    
    def __init__(self, in_features, hidden_dim, num_views=3, dropout=0.3):
        super(MultiViewGCN, self).__init__()
        
        self.num_views = num_views
        self.dropout = dropout
        
        # Separate GCN for each view
        self.view_gcns = nn.ModuleList([
            GraphConvolution(in_features, hidden_dim)
            for _ in range(num_views)
        ])
        
        # Layer normalization for each view
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_views)
        ])
    
    def forward(self, x, graphs):
        """
        Forward pass through all views
        
        Args:
            x: Input features (batch, num_nodes, in_features)
            graphs: Dictionary of adjacency matrices
                    Keys: 'physical', 'proximity', 'correlation'
        
        Returns:
            view_outputs: List of outputs from each view
        """
        view_outputs = []
        
        graph_list = [graphs['physical'], graphs['proximity'], graphs['correlation']]
        
        for i, (gcn, norm, adj) in enumerate(zip(self.view_gcns, self.layer_norms, graph_list)):
            # Graph convolution
            h = gcn(x, adj)
            
            # Layer normalization
            h = norm(h)
            
            # Activation
            h = F.relu(h)
            
            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            view_outputs.append(h)
        
        return view_outputs


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multiple graph views
    
    Args:
        hidden_dim: Dimension of view features
        num_views: Number of views to fuse
    """
    
    def __init__(self, hidden_dim, num_views=3):
        super(AttentionFusion, self).__init__()
        
        self.num_views = num_views
        self.hidden_dim = hidden_dim
        
        # Attention weights for each view
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, view_outputs):
        """
        Fuse multiple views using attention
        
        Args:
            view_outputs: List of tensors (batch, num_nodes, hidden_dim)
        
        Returns:
            fused: Fused features (batch, num_nodes, hidden_dim)
        """
        # Stack views: (num_views, batch, num_nodes, hidden_dim)
        stacked = torch.stack(view_outputs, dim=0)
        
        # Calculate attention scores for each view
        # (num_views, batch, num_nodes, 1)
        attention_scores = self.attention(stacked)
        
        # Softmax over views
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Weighted sum: (batch, num_nodes, hidden_dim)
        fused = torch.sum(attention_weights * stacked, dim=0)
        
        return fused


class TemporalModule(nn.Module):
    """
    Temporal module using GRU for time series modeling
    
    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of GRU layers
        dropout: Dropout rate
    """
    
    def __init__(self, hidden_dim, num_layers=2, dropout=0.3):
        super(TemporalModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Forward pass through temporal module
        
        Args:
            x: Input features (batch, seq_length, num_nodes, hidden_dim)
        
        Returns:
            output: Temporal features (batch, num_nodes, hidden_dim)
        """
        batch_size, seq_length, num_nodes, hidden_dim = x.size()
        
        # Reshape to process all nodes together
        # (batch * num_nodes, seq_length, hidden_dim)
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.view(batch_size * num_nodes, seq_length, hidden_dim)
        
        # Apply GRU
        output, _ = self.gru(x_reshaped)
        
        # Take last timestep
        output = output[:, -1, :]  # (batch * num_nodes, hidden_dim)
        
        # Reshape back
        output = output.view(batch_size, num_nodes, hidden_dim)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


if __name__ == '__main__':
    print("Testing GCN Layers...")
    
    # Test parameters
    batch_size = 8
    num_nodes = 207
    in_features = 1
    hidden_dim = 64
    seq_length = 12
    
    # Create dummy data
    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj > 0.95).float()  # Sparse graph
    adj = adj + adj.T  # Symmetric
    
    print(f"\n✓ Created test data:")
    print(f"  X shape: {x.shape}")
    print(f"  Adj shape: {adj.shape}")
    print(f"  Adj density: {adj.sum() / (num_nodes ** 2):.4f}")
    
    # Test GraphConvolution
    print("\n🧪 Testing GraphConvolution layer...")
    gcn = GraphConvolution(in_features, hidden_dim)
    out = gcn(x, adj)
    print(f"✓ Output shape: {out.shape}")
    
    # Test MultiViewGCN
    print("\n🧪 Testing MultiViewGCN...")
    multi_gcn = MultiViewGCN(in_features, hidden_dim, num_views=3)
    graphs = {'physical': adj, 'proximity': adj, 'correlation': adj}
    view_outputs = multi_gcn(x, graphs)
    print(f"✓ Number of views: {len(view_outputs)}")
    print(f"✓ Each view shape: {view_outputs[0].shape}")
    
    # Test AttentionFusion
    print("\n🧪 Testing AttentionFusion...")
    fusion = AttentionFusion(hidden_dim, num_views=3)
    fused = fusion(view_outputs)
    print(f"✓ Fused shape: {fused.shape}")
    
    # Test TemporalModule
    print("\n🧪 Testing TemporalModule...")
    temporal = TemporalModule(hidden_dim, num_layers=2)
    x_temporal = torch.randn(batch_size, seq_length, num_nodes, hidden_dim)
    temporal_out = temporal(x_temporal)
    print(f"✓ Temporal output shape: {temporal_out.shape}")
    
    print("\n✅ All layer tests passed!")
