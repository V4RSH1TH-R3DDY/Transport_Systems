# Traffic Datasets for TRAF-GNN

## Available Datasets

We'll use the **METR-LA** dataset as our primary benchmark.

### 1. METR-LA (Los Angeles Metro)
- **Sensors**: 207 loop detectors
- **Time Period**: March 1, 2012 - June 30, 2012 (4 months)
- **Frequency**: 5-minute intervals
- **Metric**: Traffic speed (mph)
- **Size**: ~34,000 timesteps

### 2. PeMS-BAY (Bay Area)
- **Sensors**: 325 loop detectors  
- **Time Period**: January 1, 2017 - May 31, 2017 (5 months)
- **Frequency**: 5-minute intervals
- **Metric**: Traffic speed (mph)
- **Size**: ~52,000 timesteps

## Download Sources

### Recommended: METR-LA from GitHub

We'll use the preprocessed METR-LA dataset from multiple reliable sources:

1. **Primary**: [hazdzz/dcrnn_data](https://github.com/hazdzz/dcrnn_data)
2. **Backup**: [deepkashiwa20/DL-Traff-Graph](https://github.com/deepkashiwa20/DL-Traff-Graph)
3. **Zenodo**: [CSV Format](https://zenodo.org/records/5724362)

## Data Format

The dataset typically includes:
- `metr-la.h5` - Traffic speed time series (207 sensors × 34,272 timesteps)
- `adj_mx.pkl` - Road network adjacency matrix (207 × 207)
- `graph_sensor_ids.txt` - Sensor IDs
- `graph_sensor_locations.csv` - Latitude/longitude coordinates

## Download Instructions

Run the download script:
```bash
python src/download_data.py --dataset metr-la
```

Or manually download from:
```
https://github.com/hazdzz/dcrnn_data/tree/main/data/METR-LA
```

## License & Citation

This data is derived from Caltrans Performance Measurement System (PeMS) and is openly available for research purposes.

**Original Paper**:
```
Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). 
Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. 
ICLR 2018.
```

## Next Steps

After downloading:
1. Verify data integrity
2. Explore in `notebooks/01_data_exploration.ipynb`
3. Preprocess using `src/preprocessing.py`
