"""
PyTorch Dataset for TRAF-GNN
Loads preprocessed traffic data and multi-view graphs
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class TrafficDataset(Dataset):
    """
    Traffic forecasting dataset with multi-view graphs
    
    Args:
        data_dir: Directory containing processed data
        graph_dir: Directory containing graph adjacency matrices
        dataset_name: Name of dataset (e.g., 'metr-la')
        split: 'train', 'val', or 'test'
        use_demo_graphs: If True, use demo graphs (207 nodes), else use real graphs (4106 nodes)
    """
    
    def __init__(self, data_dir='data/processed', graph_dir='graphs', 
                 dataset_name='metr-la', split='train', use_demo_graphs=True):
        self.data_dir = Path(data_dir)
        self.graph_dir = Path(graph_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.use_demo_graphs = use_demo_graphs
        
        # Load data
        self.X, self.y = self._load_data()
        
        # Load graphs
        self.graphs = self._load_graphs()
        
        print(f"✓ Loaded {split} dataset:")
        print(f"  Samples: {len(self)}")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Target shape: {self.y.shape}")
        print(f"  Graphs: {len(self.graphs)} views, shape {list(self.graphs.values())[0].shape}")
    
    def _load_data(self):
        """Load preprocessed time series data"""
        X_file = self.data_dir / f'{self.dataset_name}_X_{self.split}.npy'
        y_file = self.data_dir / f'{self.dataset_name}_y_{self.split}.npy'
        
        if X_file.exists() and y_file.exists():
            X = np.load(X_file)
            y = np.load(y_file)
        else:
            # Generate dummy data for testing
            print(f"⚠️  Data files not found, creating dummy data for testing...")
            num_samples = 1000 if self.split == 'train' else 200
            num_nodes = 207 if self.use_demo_graphs else 4106
            seq_length = 12
            pred_horizon = 3
            
            X = np.random.randn(num_samples, seq_length, num_nodes).astype(np.float32)
            y = np.random.randn(num_samples, pred_horizon, num_nodes).astype(np.float32)
        
        return X, y
    
    def _load_graphs(self):
        """Load multi-view graph adjacency matrices"""
        prefix = '' if self.use_demo_graphs else 'real_'
        
        graph_files = {
            'physical': self.graph_dir / f'{prefix}A_physical.npy',
            'proximity': self.graph_dir / f'{prefix}A_proximity.npy',
            'correlation': self.graph_dir / f'{prefix}A_correlation.npy'
        }
        
        # Try demo graphs first
        if not all(f.exists() for f in graph_files.values()) and not self.use_demo_graphs:
            print(f"⚠️  Real graphs not found, falling back to demo graphs...")
            self.use_demo_graphs = True
            graph_files = {
                'physical': self.graph_dir / 'metr-la_A_physical.npy',
                'proximity': self.graph_dir / 'metr-la_A_proximity.npy',
                'correlation': self.graph_dir / 'metr-la_A_correlation.npy'
            }
        
        graphs = {}
        for view_name, file_path in graph_files.items():
            if file_path.exists():
                adj = np.load(file_path)
                graphs[view_name] = torch.FloatTensor(adj)
            else:
                # Create dummy graph
                num_nodes = self.X.shape[2]
                print(f"⚠️  {view_name} graph not found, creating dummy graph...")
                adj = np.eye(num_nodes, dtype=np.float32)
                graphs[view_name] = torch.FloatTensor(adj)
        
        return graphs
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Input sequence (seq_length, num_nodes)
            y: Target sequence (pred_horizon, num_nodes)
            graphs: Dictionary of adjacency matrices
        """
        x = torch.FloatTensor(self.X[idx])
        y_target = torch.FloatTensor(self.y[idx])
        
        return x, y_target, self.graphs


def create_dataloaders(data_dir='data/processed', graph_dir='graphs',
                       dataset_name='metr-la', batch_size=32, 
                       use_demo_graphs=True, num_workers=0):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory with preprocessed data
        graph_dir: Directory with graph adjacency matrices  
        dataset_name: Name of dataset
        batch_size: Batch size for training
        use_demo_graphs: Use 207-node demo graphs (True) or 4106-node real graphs (False)
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    print("\n" + "="*60)
    print("🔧 Creating Data Loaders")
    print("="*60)
    
    # Create datasets
    train_dataset = TrafficDataset(
        data_dir, graph_dir, dataset_name, 'train', use_demo_graphs
    )
    val_dataset = TrafficDataset(
        data_dir, graph_dir, dataset_name, 'val', use_demo_graphs
    )
    test_dataset = TrafficDataset(
        data_dir, graph_dir, dataset_name, 'test', use_demo_graphs
    )
    
    # Custom collate function to handle graphs
    def collate_fn(batch):
        """Collate function that handles graph dictionaries"""
        xs, ys, graphs = zip(*batch)
        
        # Stack sequences
        x_batch = torch.stack(xs)
        y_batch = torch.stack(ys)
        
        # Graphs are the same for all samples, just use first one
        graphs_batch = graphs[0]
        
        return x_batch, y_batch, graphs_batch
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\n✅ Data Loaders Created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Using: {'Demo graphs (207 nodes)' if use_demo_graphs else 'Real graphs (4106 nodes)'}")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    print("Testing Traffic Dataset...")
    
    # Create loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=16,
        use_demo_graphs=True
    )
    
    # Test one batch
    print("\n🧪 Testing one batch...")
    for x_batch, y_batch, graphs in train_loader:
        print(f"\n✓ Batch loaded successfully:")
        print(f"  X shape: {x_batch.shape}  # (batch, seq_length, num_nodes)")
        print(f"  Y shape: {y_batch.shape}  # (batch, pred_horizon, num_nodes)")
        print(f"  Graphs: {list(graphs.keys())}")
        print(f"  Physical graph shape: {graphs['physical'].shape}")
        break
    
    print("\n✅ Dataset test passed!")
