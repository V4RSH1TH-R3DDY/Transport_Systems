"""
Dataset Download Utility for TRAF-GNN
Downloads METR-LA and PeMS-BAY traffic datasets
"""

import os
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import shutil


def download_file(url, destination, desc="Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_metr_la(data_dir):
    """Download METR-LA dataset from GitHub"""
    print("\n📥 Downloading METR-LA dataset...")
    
    base_url = "https://github.com/deepkashiwa20/DL-Traff-Graph/raw/main/data/METR-LA/"
    files = [
        "metr-la.h5",
        "adj_mx.pkl",
        "graph_sensor_ids.txt",
        "graph_sensor_locations.csv"
    ]
    
    for filename in files:
        url = base_url + filename
        dest = data_dir / filename
        
        if dest.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
            
        try:
            print(f"\nDownloading {filename}...")
            download_file(url, dest, desc=filename)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print(f"  Please manually download from: {url}")
    
    print("\n✅ METR-LA dataset download complete!")


def download_pems_bay(data_dir):
    """Download PeMS-BAY dataset from GitHub"""
    print("\n📥 Downloading PeMS-BAY dataset...")
    
    base_url = "https://github.com/deepkashiwa20/DL-Traff-Graph/raw/main/data/PEMS-BAY/"
    files = [
        "pems-bay.h5",
        "adj_mx.pkl",
        "graph_sensor_ids.txt",
        "graph_sensor_locations.csv"
    ]
    
    for filename in files:
        url = base_url + filename
        dest = data_dir / filename
        
        if dest.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
            
        try:
            print(f"\nDownloading {filename}...")
            download_file(url, dest, desc=filename)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print(f"  Please manually download from: {url}")
    
    print("\n✅ PeMS-BAY dataset download complete!")


def verify_dataset(data_dir, dataset_name):
    """Verify downloaded dataset files"""
    print(f"\n🔍 Verifying {dataset_name} dataset...")
    
    required_files = {
        'metr-la': ['metr-la.h5', 'adj_mx.pkl', 'graph_sensor_ids.txt', 'graph_sensor_locations.csv'],
        'pems-bay': ['pems-bay.h5', 'adj_mx.pkl', 'graph_sensor_ids.txt', 'graph_sensor_locations.csv']
    }
    
    files = required_files.get(dataset_name, [])
    all_present = True
    
    for filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {filename} - MISSING")
            all_present = False
    
    if all_present:
        print(f"\n✅ All {dataset_name} files verified!")
    else:
        print(f"\n⚠️  Some {dataset_name} files are missing. Please check downloads.")
    
    return all_present


def main():
    parser = argparse.ArgumentParser(description='Download traffic datasets for TRAF-GNN')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['metr-la', 'pems-bay', 'both'],
        default='metr-la',
        help='Dataset to download (default: metr-la)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory to save downloaded files (default: data/raw)'
    )
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚦 TRAF-GNN Dataset Downloader")
    print("=" * 60)
    
    # Download datasets
    if args.dataset in ['metr-la', 'both']:
        download_metr_la(data_dir)
        verify_dataset(data_dir, 'metr-la')
    
    if args.dataset in ['pems-bay', 'both']:
        download_pems_bay(data_dir)
        verify_dataset(data_dir, 'pems-bay')
    
    print("\n" + "=" * 60)
    print("📊 Next Steps:")
    print("  1. Explore the data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("  2. Preprocess data: python src/preprocessing.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
