"""
Data Preprocessing Pipeline for TRAF-GNN
Handles missing values, normalization, and train/val/test splits
"""

import numpy as np
import pandas as pd
import h5py
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import argparse
import json


class TrafficDataPreprocessor:
    """Preprocesses traffic data for TRAF-GNN model"""
    
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.data_stats = {}
        
    def load_data(self, dataset='metr-la'):
        """Load raw traffic data"""
        print(f"\n📥 Loading {dataset.upper()} dataset...")
        
        if dataset == 'metr-la':
            h5_file = self.raw_data_dir / 'metr-la.h5'
        elif dataset == 'pems-bay':
            h5_file = self.raw_data_dir / 'pems-bay.h5'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Load traffic data
        with h5py.File(h5_file, 'r') as f:
            # Try common key names
            for key in ['speed', 'data', 'df']:
                if key in f.keys():
                    data = f[key][:]
                    break
            else:
                # Use first key if none match
                data = f[list(f.keys())[0]][:]
        
        print(f"✓ Loaded data shape: {data.shape}")
        print(f"  Timesteps: {data.shape[0]:,}")
        print(f"  Sensors: {data.shape[1]}")
        
        # Load adjacency matrix
        adj_file = self.raw_data_dir / 'adj_mx.pkl'
        with open(adj_file, 'rb') as f:
            try:
                sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
            except:
                # Alternative unpacking if structure is different
                pickle_data = pickle.load(f, encoding='latin1')
                adj_mx = pickle_data[2] if len(pickle_data) == 3 else pickle_data
                sensor_ids = None
        
        print(f"✓ Loaded adjacency matrix shape: {adj_mx.shape}")
        
        return data, adj_mx, sensor_ids
    
    def handle_missing_values(self, data, method='linear'):
        """Handle missing values in traffic data
        
        Args:
            data: numpy array of shape (timesteps, sensors)
            method: 'linear', 'forward', 'backward', or 'mean'
        """
        print(f"\n🔧 Handling missing values (method: {method})...")
        
        initial_missing = np.isnan(data).sum()
        initial_pct = (initial_missing / data.size) * 100
        print(f"  Initial missing: {initial_missing:,} ({initial_pct:.2f}%)")
        
        data_filled = data.copy()
        
        if method == 'linear':
            # Linear interpolation along time axis
            df = pd.DataFrame(data)
            df_interpolated = df.interpolate(method='linear', axis=0, limit_direction='both')
            data_filled = df_interpolated.values
            
        elif method == 'forward':
            df = pd.DataFrame(data)
            data_filled = df.fillna(method='ffill').fillna(method='bfill').values
            
        elif method == 'backward':
            df = pd.DataFrame(data)
            data_filled = df.fillna(method='bfill').fillna(method='ffill').values
            
        elif method == 'mean':
            # Fill with column mean
            col_means = np.nanmean(data, axis=0)
            for i in range(data.shape[1]):
                mask = np.isnan(data[:, i])
                data_filled[mask, i] = col_means[i]
        
        remaining_missing = np.isnan(data_filled).sum()
        print(f"✓ Remaining missing: {remaining_missing:,}")
        
        # Fill any remaining NaNs with 0
        if remaining_missing > 0:
            print(f"  Filling {remaining_missing} remaining NaNs with 0")
            data_filled = np.nan_to_num(data_filled, nan=0.0)
        
        return data_filled
    
    def normalize_data(self, data, method='zscore'):
        """Normalize traffic data
        
        Args:
            data: numpy array of shape (timesteps, sensors)
            method: 'zscore' or 'minmax'
        """
        print(f"\n📊 Normalizing data (method: {method})...")
        
        if method == 'zscore':
            # Z-score normalization
            data_normalized = self.scaler.fit_transform(data)
            
            self.data_stats['mean'] = self.scaler.mean_
            self.data_stats['std'] = self.scaler.scale_
            
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            data_normalized = (data - data_min) / (data_max - data_min + 1e-8)
            
            self.data_stats['min'] = data_min
            self.data_stats['max'] = data_max
        
        print(f"✓ Normalized data - mean: {np.mean(data_normalized):.4f}, std: {np.std(data_normalized):.4f}")
        
        return data_normalized
    
    def create_sequences(self, data, seq_length=12, pred_horizon=3):
        """Create input-output sequences for time series prediction
        
        Args:
            data: normalized data (timesteps, sensors)
            seq_length: number of historical timesteps to use
            pred_horizon: number of future timesteps to predict
        """
        print(f"\n🔄 Creating sequences (seq_len={seq_length}, pred_horizon={pred_horizon})...")
        
        X, y = [], []
        
        for i in range(len(data) - seq_length - pred_horizon + 1):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length:i+seq_length+pred_horizon])
        
        X = np.array(X)  # Shape: (num_samples, seq_length, num_sensors)
        y = np.array(y)  # Shape: (num_samples, pred_horizon, num_sensors)
        
        print(f"✓ Created sequences:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        return X, y
    
    def train_val_test_split(self, X, y, train_ratio=0.7, val_ratio=0.1):
        """Split data into train/validation/test sets (temporal split)"""
        print(f"\n✂️  Splitting data (train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio})...")
        
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"✓ Split sizes:")
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Val:   {len(X_val):,} samples")
        print(f"  Test:  {len(X_test):,} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_processed_data(self, train_data, val_data, test_data, adj_mx, dataset_name='metr-la'):
        """Save processed data to disk"""
        print(f"\n💾 Saving processed data...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Save as numpy arrays
        np.save(self.processed_data_dir / f'{dataset_name}_X_train.npy', X_train)
        np.save(self.processed_data_dir / f'{dataset_name}_y_train.npy', y_train)
        np.save(self.processed_data_dir / f'{dataset_name}_X_val.npy', X_val)
        np.save(self.processed_data_dir / f'{dataset_name}_y_val.npy', y_val)
        np.save(self.processed_data_dir / f'{dataset_name}_X_test.npy', X_test)
        np.save(self.processed_data_dir / f'{dataset_name}_y_test.npy', y_test)
        
        # Save adjacency matrix
        np.save(self.processed_data_dir / f'{dataset_name}_adj_mx.npy', adj_mx)
        
        # Save normalization statistics
        with open(self.processed_data_dir / f'{dataset_name}_stats.json', 'w') as f:
            stats_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in self.data_stats.items()}
            json.dump(stats_serializable, f, indent=2)
        
        print(f"✓ Saved all processed files to {self.processed_data_dir}")
        
        # Print file sizes
        for file in self.processed_data_dir.glob(f'{dataset_name}*'):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.2f} MB")
    
    def process(self, dataset='metr-la', seq_length=12, pred_horizon=3, 
                missing_method='linear', norm_method='zscore'):
        """Complete preprocessing pipeline"""
        print("=" * 60)
        print("🚦 TRAF-GNN Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Load data
        data, adj_mx, sensor_ids = self.load_data(dataset)
        
        # Handle missing values
        data_filled = self.handle_missing_values(data, method=missing_method)
        
        # Normalize
        data_normalized = self.normalize_data(data_filled, method=norm_method)
        
        # Create sequences
        X, y = self.create_sequences(data_normalized, seq_length, pred_horizon)
        
        # Split data
        train_data, val_data, test_data = self.train_val_test_split(X, y)
        
        # Save
        self.save_processed_data(train_data, val_data, test_data, adj_mx, dataset)
        
        print("\n" + "=" * 60)
        print("✅ Preprocessing complete!")
        print("=" * 60)
        print(f"\n📋 Processed Data Summary:")
        print(f"  Dataset: {dataset.upper()}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Prediction horizon: {pred_horizon}")
        print(f"  Sensors: {data.shape[1]}")
        print(f"  Train samples: {len(train_data[0]):,}")
        print(f"  Val samples: {len(val_data[0]):,}")
        print(f"  Test samples: {len(test_data[0]):,}")
        print("\n📊 Next Steps:")
        print("  1. Build multi-view graphs: python src/build_graphs.py")
        print("  2. Train model: python src/train.py")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess traffic data for TRAF-GNN')
    parser.add_argument('--dataset', type=str, default='metr-la', 
                       choices=['metr-la', 'pems-bay'],
                       help='Dataset to preprocess')
    parser.add_argument('--seq-length', type=int, default=12,
                       help='Input sequence length (default: 12 = 1 hour)')
    parser.add_argument('--pred-horizon', type=int, default=3,
                       help='Prediction horizon (default: 3 = 15 minutes)')
    parser.add_argument('--missing-method', type=str, default='linear',
                       choices=['linear', 'forward', 'backward', 'mean'],
                       help='Method for handling missing values')
    parser.add_argument('--norm-method', type=str, default='zscore',
                       choices=['zscore', 'minmax'],
                       help='Normalization method')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocessor = TrafficDataPreprocessor()
    preprocessor.process(
        dataset=args.dataset,
        seq_length=args.seq_length,
        pred_horizon=args.pred_horizon,
        missing_method=args.missing_method,
        norm_method=args.norm_method
    )


if __name__ == '__main__':
    main()
