"""
Build adjacency matrix from distances CSV file
This creates a proper adjacency matrix when pickle file is corrupted
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

print("="*60)
print("🔧 Building Adjacency Matrix from Distances")
print("="*60)

data_dir = Path('data/raw')

# Load distances CSV
print("\n📊 Loading distances CSV...")
distances_file = data_dir / 'distances_la_2012.csv'

if distances_file.exists():
    df = pd.read_csv(distances_file)
    print(f"✓ Loaded distances: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(df.head())
    
    # Extract unique sensor IDs
    if 'from' in df.columns and 'to' in df.columns:
        from_sensors = df['from'].unique()
        to_sensors = df['to'].unique()
        all_sensors = sorted(set(list(from_sensors) + list(to_sensors)))
        num_sensors = len(all_sensors)
        
        print(f"\n🗺️  Found {num_sensors} unique sensors")
        
        # Create mapping
        sensor_to_idx = {sensor: idx for idx, sensor in enumerate(all_sensors)}
        
        # Build adjacency matrix
        print("\n🏗️  Building adjacency matrix...")
        adj_mx = np.zeros((num_sensors, num_sensors))
        
        # Use distance threshold to define connectivity
        # Sensors within ~2km are considered connected
        threshold_km = 2.0
        
        for _, row in df.iterrows():
            from_idx = sensor_to_idx[row['from']]
            to_idx = sensor_to_idx[row['to']]
            distance = row['cost']  # distance in km
            
            if distance <= threshold_km:
                # Weight by inverse distance (closer = stronger connection)
                weight = np.exp(-distance / 0.5)  # Gaussian kernel
                adj_mx[from_idx, to_idx] = weight
                adj_mx[to_idx, from_idx] = weight  # Make symmetric
        
        # Ensure self-loops
        np.fill_diagonal(adj_mx, 1.0)
        
        print(f"✓ Created adjacency matrix: {adj_mx.shape}")
        print(f"  Total edges (threshold={threshold_km}km): {np.sum(adj_mx > 0)}")
        print(f"  Density: {np.sum(adj_mx > 0) / (num_sensors ** 2):.4f}")
        print(f"  Average degree: {np.mean(np.sum(adj_mx > 0, axis=1)):.2f}")
        
        # Save as proper pickle
        print("\n💾 Saving corrected adjacency matrix...")
        sensor_ids = all_sensors
        sensor_id_to_ind = sensor_to_idx
        
        # Save in the format expected by build_graphs.py
        with open(data_dir / 'adj_mx_corrected.pkl', 'wb') as f:
            pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f, protocol=2)
        
        # Also save as numpy for easy loading
        np.save(data_dir / 'adj_mx_corrected.npy', adj_mx)
        
        print(f"✓ Saved to:")
        print(f"  - adj_mx_corrected.pkl (pickle format)")
        print(f"  - adj_mx_corrected.npy (numpy format)")
        
        # Verify
        print("\n🔍 Verifying saved file...")
        with open(data_dir / 'adj_mx_corrected.pkl', 'rb') as f:
            loaded = pickle.load(f)
            print(f"✓ Successfully loaded back")
            print(f"  Adjacency matrix shape: {loaded[2].shape}")
        
else:
    print(f"⚠️  Distances file not found: {distances_file}")
    print("   Creating minimal synthetic adjacency matrix...")
    
    # Create minimal adjacency for 207 sensors (METR-LA size)
    num_sensors = 207
    adj_mx = np.zeros((num_sensors, num_sensors))
    
    # Connect each node to 3-4 neighbors (realistic for road network)
    for i in range(num_sensors):
        # Connect to next few nodes (simulating road connections)
        neighbors = [(i+1) % num_sensors, (i+2) % num_sensors, (i-1) % num_sensors]
        for j in neighbors:
            adj_mx[i, j] = 1.0
            adj_mx[j, i] = 1.0
    
    np.fill_diagonal(adj_mx, 1.0)
    
    sensor_ids = [str(i) for i in range(num_sensors)]
    sensor_id_to_ind = {str(i): i for i in range(num_sensors)}
    
    with open(data_dir / 'adj_mx_corrected.pkl', 'wb') as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f, protocol=2)
    
    print(f"✓ Created synthetic adjacency: {adj_mx.shape}")

print("\n" + "="*60)
print("✅ Adjacency Matrix Ready!")
print("="*60)
print("\n📋 Next: Update build_graphs.py to use 'adj_mx_corrected.pkl'")
print("="*60)
