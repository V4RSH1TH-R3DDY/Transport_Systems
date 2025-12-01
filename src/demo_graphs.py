"""
Quick Demo: Generate Multi-View Graphs with Synthetic Data
This creates demonstration graphs to show the multi-view concept working
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup
graphs_dir = Path('graphs')
graphs_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🚦 Generating Demo Multi-View Graphs")
print("=" * 60)

# Create synthetic data for demonstration
num_nodes = 207  # METR-LA size
np.random.seed(42)

# 1. Physical Topology Graph (sparse, road connections)
print("\n📐 Creating Physical Topology Graph...")
A_physical = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    # Connect to 2-3 neighbors (simulating road network)
    num_neighbors = np.random.randint(2, 4)
    neighbors = np.random.choice(num_nodes, num_neighbors, replace=False)
    A_physical[i, neighbors] = 1
# Make symmetric
A_physical = np.maximum(A_physical, A_physical.T)
# Add self-loops
np.fill_diagonal(A_physical, 1.0)
print(f"✓ Physical graph: {np.sum(A_physical > 0)} edges, density: {np.sum(A_physical > 0) / (num_nodes ** 2):.4f}")

# 2. Spatial Proximity Graph (k-NN, denser)
print("\n🌍 Creating Spatial Proximity Graph (k=10)...")
A_proximity = np.zeros((num_nodes, num_nodes))
k = 10
for i in range(num_nodes):
    # Assign random distances and select k-nearest
    distances = np.random.rand(num_nodes)
    distances[i] = -1  # Self
    nearest_k = np.argsort(distances)[:k]
    A_proximity[i, nearest_k] = 1
# Make symmetric
A_proximity = np.maximum(A_proximity, A_proximity.T)
# Add self-loops
np.fill_diagonal(A_proximity, 1.0)
print(f"✓ Proximity graph: {np.sum(A_proximity > 0)} edges, density: {np.sum(A_proximity > 0) / (num_nodes ** 2):.4f}")

# 3. Traffic Correlation Graph (pattern-based)
print("\n📊 Creating Traffic Correlation Graph (k=10)...")
A_correlation = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    # Simulate correlation-based connections
    correlations = np.random.rand(num_nodes)
    correlations[i] = 0  # Self
    top_k = np.argsort(correlations)[-k:]
    A_correlation[i, top_k] = 1
# Make symmetric
A_correlation = np.maximum(A_correlation, A_correlation.T)
# Add self-loops
np.fill_diagonal(A_correlation, 1.0)
print(f"✓ Correlation graph: {np.sum(A_correlation > 0)} edges, density: {np.sum(A_correlation > 0) / (num_nodes ** 2):.4f}")

# Save graphs
print("\n💾 Saving graphs...")
np.save(graphs_dir / 'metr-la_A_physical.npy', A_physical.astype(np.float32))
np.save(graphs_dir / 'metr-la_A_proximity.npy', A_proximity.astype(np.float32))
np.save(graphs_dir / 'metr-la_A_correlation.npy', A_correlation.astype(np.float32))
print(f"✓ Saved 3 graph files to {graphs_dir}/")

# Visualize
print("\n📊 Creating visualization...")
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
plt.savefig(graphs_dir / 'graph_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to {graphs_dir}/graph_comparison.png")
plt.close()

# Analyze overlaps
print("\n🔍 Graph Overlap Analysis:")
phys_edges = set(zip(*np.where(A_physical > 0)))
prox_edges = set(zip(*np.where(A_proximity > 0)))
corr_edges = set(zip(*np.where(A_correlation > 0)))

overlap_phys_prox = len(phys_edges & prox_edges)
overlap_phys_corr = len(phys_edges & corr_edges)
overlap_prox_corr = len(prox_edges & corr_edges)
overlap_all = len(phys_edges & prox_edges & corr_edges)

print(f"  Physical ∩ Proximity: {overlap_phys_prox} edges")
print(f"  Physical ∩ Correlation: {overlap_phys_corr} edges")
print(f"  Proximity ∩ Correlation: {overlap_prox_corr} edges")
print(f"  All three: {overlap_all} edges")

print("\n" + "=" * 60)
print("✅ Demo Multi-View Graphs Created!")
print("=" * 60)
print("\n📋 Summary:")
print(f"  Nodes: {num_nodes}")
print(f"  Physical graph edges: {np.sum(A_physical > 0)}")
print(f"  Proximity graph edges: {np.sum(A_proximity > 0)}")
print(f"  Correlation graph edges: {np.sum(A_correlation > 0)}")
print(f"\n📊 Files created in graphs/:")
print(f"  - metr-la_A_physical.npy")
print(f"  - metr-la_A_proximity.npy")
print(f"  - metr-la_A_correlation.npy")
print(f"  - graph_comparison.png")
print("\n✨ Phase 2 demonstration complete!")
print("   (Note: These are synthetic graphs for demonstration)")
print("   (To use real data, fix the pickle file download)")
print("=" * 60)
