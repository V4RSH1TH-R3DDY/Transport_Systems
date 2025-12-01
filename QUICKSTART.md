# 🚀 Quick Start Guide

## Phase 1: Data Collection & Preprocessing

Follow these steps to set up your environment and preprocess the data:

### Step 1: Set Up Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset

```powershell
# Download METR-LA dataset (recommended for first run)
python src/download_data.py --dataset metr-la

# Or download both datasets
python src/download_data.py --dataset both
```

**Expected Output:**
- `data/raw/metr-la.h5` - Traffic speed data
- `data/raw/adj_mx.pkl` - Road network adjacency matrix
- `data/raw/graph_sensor_ids.txt` - Sensor IDs
- `data/raw/graph_sensor_locations.csv` - Sensor coordinates

### Step 3: Explore the Data (Optional but Recommended)

```powershell
# Launch Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

**What you'll explore:**
- Data quality and missing values
- Traffic speed distributions
- Daily/hourly patterns
- Spatial network structure
- Sensor correlations

### Step 4: Preprocess Data

```powershell
# Run preprocessing with default parameters
python src/preprocessing.py --dataset metr-la

# Or customize parameters
python src/preprocessing.py --dataset metr-la \
  --seq-length 12 \     # 1 hour of history (12 x 5-min intervals)
  --pred-horizon 3 \    # Predict 15 minutes ahead (3 x 5-min)
  --missing-method linear \  # Linear interpolation
  --norm-method zscore       # Z-score normalization
```

**Expected Output:**
```
data/processed/
  ├── metr-la_X_train.npy       # Training inputs
  ├── metr-la_y_train.npy       # Training targets
  ├── metr-la_X_val.npy         # Validation inputs
  ├── metr-la_y_val.npy         # Validation targets
  ├── metr-la_X_test.npy        # Test inputs
  ├── metr-la_y_test.npy        # Test targets
  ├── metr-la_adj_mx.npy        # Adjacency matrix
  └── metr-la_stats.json        # Normalization statistics
```

### Step 5: Verify Preprocessing

Check that all files are created:

```powershell
dir data\processed\
```

You should see 8 files:
- 6 data files (X_train, y_train, X_val, y_val, X_test, y_test)
- 1 adjacency matrix
- 1 stats file

---

## Understanding the Data

### Data Shapes

After preprocessing with default parameters:

```
X_train: (N_train, 12, 207)  # N_train sequences, 12 timesteps, 207 sensors
y_train: (N_train, 3, 207)   # N_train sequences, 3 future timesteps, 207 sensors
```

- **Input (X)**: Last 12 timesteps (1 hour) of traffic speeds from all sensors
- **Target (y)**: Next 3 timesteps (15 minutes) to predict

### Parameter Guide

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `--seq-length` | Historical window | 12 (1 hour) | 12-24 |
| `--pred-horizon` | Prediction steps | 3 (15 min) | 3-12 |
| `--missing-method` | Fill NaNs | linear | linear or forward |
| `--norm-method` | Normalization | zscore | zscore recommended |

---

## Next Steps

After completing Phase 1:

✅ **Phase 1 Complete** - Data ready for model training!

**Phase 2: Build Multi-View Graphs**
```powershell
python src/build_graphs.py
```

This will create:
- Physical topology graph
- Spatial proximity graph (k-NN)
- Traffic correlation graph

---

## Troubleshooting

### Issue: Download fails

**Solution**: Manually download from GitHub:
1. Visit: https://github.com/deepkashiwa20/DL-Traff-Graph/tree/main/data/METR-LA
2. Download all files to `data/raw/`

### Issue: Missing h5py or other package

**Solution**: Install individually:
```powershell
pip install h5py
pip install torch torch-geometric
```

### Issue: CUDA/GPU errors

**Solution**: PyTorch CPU version:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Project Status

| Phase | Status |
|-------|--------|
| ✅ Phase 0: Setup | Complete |
| ✅ Phase 1: Data Preprocessing | Complete |
| ⏳ Phase 2: Multi-View Graphs | Ready to start |
| ⏳ Phase 3: Model Architecture | Pending |

---

<div align="center">
  <strong>📊 Your data is ready! Time to build the graphs!</strong>
</div>
