"""
Fixed Dataset Downloader for TRAF-GNN
Downloads METR-LA dataset from reliable source with proper format handling
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle


def download_from_github_lfs(url, destination):
    """Download file from GitHub LFS with proper handling"""
    print(f"Downloading from GitHub: {url}")
    
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=destination.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_metr_la_fixed():
    """Download METR-LA from alternative reliable source"""
    print("\n" + "="*60)
    print("🚦 Downloading METR-LA Dataset (Fixed)")
    print("="*60)
    
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Alternative source: Use raw GitHub URLs
    base_url = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/"
    
    files_to_download = {
        'adj_mx.pkl': base_url + 'adj_mx.pkl',
        'sensor_ids.txt': base_url + 'sensor_ids.txt',
        'distances_la_2012.csv': base_url + 'distances_la_2012.csv',
    }
    
    # Download files
    for filename, url in files_to_download.items():
        dest = data_dir / filename
        if dest.exists():
            print(f"✓ {filename} already exists")
            continue
        
        try:
            download_from_github_lfs(url, dest)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"⚠️  Failed to download {filename}: {e}")
    
    # Try alternative for H5 file
    print("\n📥 Attempting to download traffic data...")
    h5_urls = [
        "https://github.com/chnsh/DCRNN_PyTorch/raw/pytorch_scratch/data/metr-la.h5",
        "https://raw.githubusercontent.com/VeritasYin/STGCN_IJCAI-18/master/data_loader/metr-la.h5",
    ]
    
    h5_dest = data_dir / 'metr-la.h5'
    if not h5_dest.exists():
        for url in h5_urls:
            try:
                print(f"Trying: {url}")
                download_from_github_lfs(url, h5_dest)
                print(f"✓ Downloaded metr-la.h5")
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue
    else:
        print(f"✓ metr-la.h5 already exists")
    
    # Verify pickle file
    print("\n🔍 Verifying adjacency matrix...")
    adj_file = data_dir / 'adj_mx.pkl'
    if adj_file.exists():
        try:
            with open(adj_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            print(f"✓ Adjacency matrix loaded successfully")
            print(f"  Type: {type(data)}")
            if isinstance(data, tuple) and len(data) >= 3:
                print(f"  Contains: sensor_ids, sensor_id_to_ind, adj_mx")
                print(f"  Adjacency shape: {data[2].shape}")
            elif isinstance(data, np.ndarray):
                print(f"  Direct array shape: {data.shape}")
        except Exception as e:
            print(f"⚠️  Pickle file verification failed: {e}")
    
    print("\n" + "="*60)
    print("✅ Download complete!")
    print("="*60)


if __name__ == '__main__':
    download_metr_la_fixed()
