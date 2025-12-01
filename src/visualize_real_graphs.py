"""
Better visualization for large real graphs - show subset
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

graphs_dir = Path('graphs')

print("=" * 60)
print("📊 Creating Better Real Graph Visualization")
print("=" * 60)

# Load real graphs
print("\n📥 Loading real graphs...")
A_physical = np.load(graphs_dir / 'real_A_physical.npy')
A_proximity = np.load(graphs_dir / 'real_A_proximity.npy')
A_correlation = np.load(graphs_dir / 'real_A_correlation.npy')

print(f"✓ Loaded graphs: {A_physical.shape}")
print(f"  Physical edges: {np.sum(A_physical > 0):,}")
print(f"  Proximity edges: {np.sum(A_proximity > 0):,}")
print(f"  Correlation edges: {np.sum(A_correlation > 0):,}")

# Visualize SUBSET (first 500 nodes for clarity)
subset_size = 500
print(f"\n🔍 Visualizing first {subset_size} nodes for clarity...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Full graphs (top row)
graphs_full = [
    (A_physical, 'Physical Topology\n(Full 4106 nodes)', 'Reds'),
    (A_proximity, 'Spatial Proximity\n(Full 4106 nodes)', 'Blues'),
    (A_correlation, 'Traffic Correlation\n(Full 4106 nodes)', 'Greens')
]

for ax, (adj, title, cmap) in zip(axes[0], graphs_full):
    im = ax.imshow(adj, cmap=cmap, aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Node ID')
    ax.set_ylabel('Node ID')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Subset graphs (bottom row) - much clearer!
graphs_subset = [
    (A_physical[:subset_size, :subset_size], f'Physical Topology\n(First {subset_size} nodes - ZOOMED)', 'Reds'),
    (A_proximity[:subset_size, :subset_size], f'Spatial Proximity\n(First {subset_size} nodes - ZOOMED)', 'Blues'),
    (A_correlation[:subset_size, :subset_size], f'Traffic Correlation\n(First {subset_size} nodes - ZOOMED)', 'Greens')
]

for ax, (adj, title, cmap) in zip(axes[1], graphs_subset):
    im = ax.imshow(adj, cmap=cmap, aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Node ID')
    ax.set_ylabel('Node ID')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add edge count
    edges = np.sum(adj > 0)
    ax.text(0.02, 0.98, f'{edges:,} edges', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(graphs_dir / 'real_graphs_detailed.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved detailed visualization")

# Create network statistics visualization
print("\n📈 Creating network statistics visualization...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

graphs = [A_physical, A_proximity, A_correlation]
titles = ['Physical Topology', 'Spatial Proximity', 'Traffic Correlation']
colors = ['red', 'blue', 'green']

for ax, adj, title, color in zip(axes, graphs, titles, colors):
    # Degree distribution
    degrees = np.sum(adj > 0, axis=1) - 1  # Exclude self-loops
    
    ax.hist(degrees, bins=50, color=color, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title}\nDegree Distribution', fontweight='bold')
    ax.axvline(np.mean(degrees), color='black', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(degrees):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(graphs_dir / 'real_graphs_statistics.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved statistics visualization")

print("\n" + "=" * 60)
print("✅ Better Visualizations Created!")
print("=" * 60)
print(f"\n📁 Files created:")
print(f"  - real_graphs_detailed.png (full + zoomed subset)")
print(f"  - real_graphs_statistics.png (degree distributions)")
print(f"\n💡 The 'empty' appearance was due to:")
print(f"  - 4,106 nodes = 16+ million pixels to plot")
print(f"  - Only 4,134 edges in physical graph (0.02% density!)")  
print(f"  - Individual connections invisible at full scale")
print(f"\n✅ Zoomed views show the structure clearly!")
print("=" * 60)
