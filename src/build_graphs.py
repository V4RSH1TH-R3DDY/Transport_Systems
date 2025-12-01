"""
Multi-View Graph Construction for TRAF-GNN
Builds three complementary graph representations:
1. Physical Topology Graph - Road network connections
2. Spatial Proximity Graph - Geographic k-NN
3. Traffic Correlation Graph - Historical pattern similarity
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from scipy.spatial.distance import cdist
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


class MultiViewGraphBuilder:
    """Constructs multiple graph views for traffic network"""
    
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed', 
                 graphs_dir='graphs'):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.graphs_dir = Path(graphs_dir)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_nodes = None
        self.sensor_locations = None
        
    def load_adjacency_matrix(self, dataset='metr-la'):
        """Load physical road network adjacency matrix"""
        print("\n📐 Loading physical topology graph...")
        
        adj_file = self.raw_data_dir / 'adj_mx.pkl'
        
        with open(adj_file, 'rb') as f:
            try:
                sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f, encoding='latin1')
                adj_mx = pickle_data[2] if len(pickle_data) == 3 else pickle_data
                sensor_ids = None
        
        self.num_nodes = adj_mx.shape[0]
        
        print(f"✓ Loaded adjacency matrix: {adj_mx.shape}")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Edges: {np.sum(adj_mx > 0)}")
        print(f"  Density: {np.sum(adj_mx > 0) / (self.num_nodes ** 2):.4f}")
        
        return adj_mx
    
    def load_sensor_locations(self, dataset='metr-la'):
        """Load sensor geographic coordinates"""
        print("\n📍 Loading sensor locations...")
        
        try:
            loc_file = self.raw_data_dir / 'graph_sensor_locations.csv'
            locations = pd.read_csv(loc_file)
            
            print(f"✓ Loaded {len(locations)} sensor locations")
            return locations
        except FileNotFoundError:
            print("⚠️  Sensor locations file not found")
            print("   Generating synthetic coordinates for demonstration...")
            # Generate random coordinates in LA area for demonstration
            np.random.seed(42)
            locations = pd.DataFrame({
                'sensor_id': range(self.num_nodes),
                'latitude': 34.05 + np.random.randn(self.num_nodes) * 0.1,
                'longitude': -118.25 + np.random.randn(self.num_nodes) * 0.1
            })
            return locations
    
    def build_physical_graph(self, adj_mx):
        """Process and save physical topology graph"""
        print("\n🏗️  Building Physical Topology Graph...")
        
        # Ensure binary adjacency (0/1)
        A_physical = (adj_mx > 0).astype(np.float32)
        
        # Add self-loops
        np.fill_diagonal(A_physical, 1.0)
        
        # Calculate statistics
        degrees = np.sum(A_physical, axis=1)
        
        print(f"✓ Physical graph statistics:")
        print(f"  Self-loops added: {self.num_nodes}")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        print(f"  Max degree: {np.max(degrees):.0f}")
        print(f"  Min degree: {np.min(degrees):.0f}")
        
        return A_physical
    
    def build_proximity_graph(self, locations, k=10, metric='haversine'):
        """Build k-NN graph based on geographic proximity"""
        print(f"\n🌍 Building Spatial Proximity Graph (k={k})...")
        
        # Extract coordinates
        if 'latitude' in locations.columns and 'longitude' in locations.columns:
            coords = locations[['latitude', 'longitude']].values
        else:
            # Try alternative column names
            coords = locations.iloc[:, [0, 1]].values
        
        # Calculate pairwise distances
        print("  Calculating pairwise distances...")
        
        if metric == 'haversine':
            # Haversine distance for lat/lon
            distances = self._haversine_distances(coords)
        else:
            # Euclidean distance
            distances = cdist(coords, coords, metric='euclidean')
        
        # Build k-NN graph
        print(f"  Selecting top-{k} nearest neighbors per node...")
        A_proximity = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        for i in tqdm(range(self.num_nodes), desc="Building k-NN graph"):
            # Get k+1 nearest (including self)
            nearest_indices = np.argsort(distances[i])[:k+1]
            A_proximity[i, nearest_indices] = 1.0
        
        # Make symmetric
        A_proximity = np.maximum(A_proximity, A_proximity.T)
        
        # Ensure self-loops
        np.fill_diagonal(A_proximity, 1.0)
        
        # Calculate statistics
        degrees = np.sum(A_proximity, axis=1)
        
        print(f"✓ Proximity graph statistics:")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        print(f"  Average distance to neighbors: {np.mean(distances[distances > 0]):.2f} km")
        
        return A_proximity
    
    def _haversine_distances(self, coords):
        """Calculate haversine distances between lat/lon coordinates"""
        lat = np.radians(coords[:, 0])
        lon = np.radians(coords[:, 1])
        
        # Expand dimensions for broadcasting
        lat1 = lat[:, np.newaxis]
        lat2 = lat[np.newaxis, :]
        lon1 = lon[:, np.newaxis]
        lon2 = lon[np.newaxis, :]
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        r = 6371
        distances = r * c
        
        return distances
    
    def build_correlation_graph(self, traffic_data, k=10, method='pearson'):
        """Build graph based on traffic correlation patterns"""
        print(f"\n📊 Building Traffic Correlation Graph (k={k}, method={method})...")
        
        # Handle missing values
        print("  Handling missing values...")
        data_filled = traffic_data.copy()
        for i in range(data_filled.shape[1]):
            col_mean = np.nanmean(data_filled[:, i])
            data_filled[np.isnan(data_filled[:, i]), i] = col_mean
        
        # Calculate correlation matrix
        print("  Calculating correlation matrix...")
        
        if method == 'pearson':
            corr_matrix = np.corrcoef(data_filled.T)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(data_filled, axis=0)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Handle NaN correlations
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Build top-k correlation graph
        print(f"  Selecting top-{k} correlated neighbors per node...")
        A_correlation = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        for i in tqdm(range(self.num_nodes), desc="Building correlation graph"):
            # Use absolute correlation and get top-k
            abs_corr = np.abs(corr_matrix[i])
            top_k_indices = np.argsort(abs_corr)[-k-1:]  # k+1 including self
            A_correlation[i, top_k_indices] = 1.0
        
        # Make symmetric
        A_correlation = np.maximum(A_correlation, A_correlation.T)
        
        # Ensure self-loops
        np.fill_diagonal(A_correlation, 1.0)
        
        # Calculate statistics
        degrees = np.sum(A_correlation, axis=1)
        avg_corr = np.mean(np.abs(corr_matrix[corr_matrix != 1.0]))
        
        print(f"✓ Correlation graph statistics:")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        print(f"  Average |correlation|: {avg_corr:.3f}")
        
        return A_correlation, corr_matrix
    
    def visualize_graphs(self, A_physical, A_proximity, A_correlation, save=True):
        """Visualize all three graph adjacency matrices"""
        print("\n📊 Generating visualizations...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        graphs = [
            (A_physical, 'Physical Topology', 'Reds'),
            (A_proximity, 'Spatial Proximity', 'Blues'),
            (A_correlation, 'Traffic Correlation', 'Greens')
        ]
        
        for ax, (adj, title, cmap) in zip(axes, graphs):
            im = ax.imshow(adj, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Node ID')
            ax.set_ylabel('Node ID')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save:
            viz_path = self.graphs_dir / 'graph_comparison.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {viz_path}")
        
        plt.close()
    
    def compare_graphs(self, A_physical, A_proximity, A_correlation):
        """Compare graph structures and overlaps"""
        print("\n🔍 Comparing graph structures...")
        
        # Edge overlaps
        physical_edges = set(zip(*np.where(A_physical > 0)))
        proximity_edges = set(zip(*np.where(A_proximity > 0)))
        correlation_edges = set(zip(*np.where(A_correlation > 0)))
        
        # Calculate overlaps
        phys_prox = len(physical_edges & proximity_edges)
        phys_corr = len(physical_edges & correlation_edges)
        prox_corr = len(proximity_edges & correlation_edges)
        all_three = len(physical_edges & proximity_edges & correlation_edges)
        
        print(f"\n📈 Edge Overlap Analysis:")
        print(f"  Physical ∩ Proximity: {phys_prox} edges")
        print(f"  Physical ∩ Correlation: {phys_corr} edges")
        print(f"  Proximity ∩ Correlation: {prox_corr} edges")
        print(f"  All three graphs: {all_three} edges")
        print(f"\n  Unique to Physical: {len(physical_edges) - phys_prox - phys_corr + all_three}")
        print(f"  Unique to Proximity: {len(proximity_edges) - phys_prox - prox_corr + all_three}")
        print(f"  Unique to Correlation: {len(correlation_edges) - phys_corr - prox_corr + all_three}")
    
    def save_graphs(self, A_physical, A_proximity, A_correlation, dataset='metr-la'):
        """Save all graph adjacency matrices"""
        print(f"\n💾 Saving graphs to {self.graphs_dir}...")
        
        np.save(self.graphs_dir / f'{dataset}_A_physical.npy', A_physical)
        np.save(self.graphs_dir / f'{dataset}_A_proximity.npy', A_proximity)
        np.save(self.graphs_dir / f'{dataset}_A_correlation.npy', A_correlation)
        
        print(f"✓ Saved 3 graph files:")
        for file in sorted(self.graphs_dir.glob(f'{dataset}_A_*.npy')):
            size_kb = file.stat().st_size / 1024
            print(f"  {file.name}: {size_kb:.2f} KB")
    
    def build_all_graphs(self, dataset='metr-la', k_proximity=10, k_correlation=10):
        """Complete graph construction pipeline"""
        print("=" * 60)
        print("🚦 TRAF-GNN Multi-View Graph Construction")
        print("=" * 60)
        
        # Load data
        adj_mx = self.load_adjacency_matrix(dataset)
        locations = self.load_sensor_locations(dataset)
        
        # Load traffic data for correlation graph
        try:
            import h5py
            h5_file = self.raw_data_dir / f'{dataset}.h5'
            with h5py.File(h5_file, 'r') as f:
                traffic_data = f['speed'][:] if 'speed' in f.keys() else f[list(f.keys())[0]][:]
            print(f"✓ Loaded traffic data: {traffic_data.shape}")
        except Exception as e:
            print(f"⚠️  Could not load traffic data for correlation graph: {e}")
            print("   Using adjacency matrix as fallback...")
            traffic_data = None
        
        # Build graphs
        A_physical = self.build_physical_graph(adj_mx)
        A_proximity = self.build_proximity_graph(locations, k=k_proximity)
        
        if traffic_data is not None:
            A_correlation, _ = self.build_correlation_graph(traffic_data, k=k_correlation)
        else:
            # Fallback: use weighted adjacency as correlation
            A_correlation = (adj_mx > 0).astype(np.float32)
            np.fill_diagonal(A_correlation, 1.0)
            print("⚠️  Using physical topology as correlation graph fallback")
        
        # Visualize and compare
        self.visualize_graphs(A_physical, A_proximity, A_correlation)
        self.compare_graphs(A_physical, A_proximity, A_correlation)
        
        # Save
        self.save_graphs(A_physical, A_proximity, A_correlation, dataset)
        
        print("\n" + "=" * 60)
        print("✅ Multi-View Graph Construction Complete!")
        print("=" * 60)
        print(f"\n📊 Summary:")
        print(f"  Dataset: {dataset.upper()}")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Graphs created: 3")
        print(f"  k_proximity: {k_proximity}")
        print(f"  k_correlation: {k_correlation}")
        print("\n📋 Next Steps:")
        print("  1. Explore graphs: jupyter notebook notebooks/02_graph_analysis.ipynb")
        print("  2. Build model: Start Phase 3 - Model Architecture")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Build multi-view graphs for TRAF-GNN')
    parser.add_argument('--dataset', type=str, default='metr-la',
                       choices=['metr-la', 'pems-bay'],
                       help='Dataset to use')
    parser.add_argument('--k-proximity', type=int, default=10,
                       help='Number of nearest neighbors for proximity graph')
    parser.add_argument('--k-correlation', type=int, default=10,
                       help='Number of most correlated neighbors for correlation graph')
    
    args = parser.parse_args()
    
    # Build graphs
    builder = MultiViewGraphBuilder()
    builder.build_all_graphs(
        dataset=args.dataset,
        k_proximity=args.k_proximity,
        k_correlation=args.k_correlation
    )


if __name__ == '__main__':
    main()
