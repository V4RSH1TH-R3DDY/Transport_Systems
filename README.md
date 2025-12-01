# 🚦 TRAF-GNN

**Multi-View Graph Learning for Traffic Forecasting with Efficient Neighbour Selection**

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📌 Overview

**TRAF-GNN** is an AI-powered traffic forecasting system that predicts future traffic flow and congestion patterns on road networks using **Graph Neural Networks (GNNs)**. Traditional traffic prediction models treat transportation data as a single graph, which often misses critical relationships like spatial proximity, topological connectivity, and historical traffic correlations.

TRAF-GNN solves this limitation by:
- **Learning from multiple graph views** simultaneously (physical topology, spatial proximity, and traffic correlation)
- **Selecting only the most relevant neighbours** for each node, reducing computational complexity
- **Combining spatial and temporal learning** for accurate spatio-temporal predictions

This project demonstrates how modern transportation systems can leverage advanced graph learning techniques to anticipate congestion, optimize routing, and improve urban mobility planning.

---

## 🎯 Key Features

### 🧠 **Multi-View Graph Architecture**
TRAF-GNN constructs and learns from three complementary graph representations:
- **Physical Road Connectivity Graph** - Captures direct road network topology
- **Spatial Proximity Graph** - Models geographic relationships between locations
- **Historical Traffic Correlation Graph** - Learns patterns from past traffic behavior

### ⚡ **Efficient Neighbour Selection**
Instead of processing all possible connections, the model intelligently selects **top-k most relevant neighbours** for each node in each view. This approach:
- Reduces computational complexity from O(n²) to O(n·k)
- Improves model focus on meaningful relationships
- Enables scalability to larger road networks

### 🔁 **Spatio-Temporal Traffic Prediction**
The architecture combines:
- **Graph Neural Networks (GNNs)** for spatial feature learning across road networks
- **Gated Recurrent Units (GRU)** / **Temporal Convolutional Networks** for time-series forecasting
- **Multi-view fusion mechanism** to aggregate insights from different graph perspectives

### 📊 **Comprehensive Visualization Tools**
Built-in visualization capabilities to:
- Plot predicted congestion patterns on interactive city maps
- Compare real vs. predicted traffic speeds with time-series charts
- Generate heatmaps showing traffic intensity across the network
- Analyze prediction errors spatially and temporally

### 🧪 **Research-Grade Foundation**
Designed for:
- Academic research and publications
- Machine learning pipeline integration
- Smart city prototypes and demonstrations
- Transportation engineering applications

---

## 🏗️ Project Structure

```
Transport_Systems/
├── data/
│   ├── raw/              # Downloaded traffic datasets (CSV, JSON, etc.)
│   └── processed/        # Preprocessed graph matrices and time-series files
│
├── graphs/
│   ├── A_physical.npy    # Adjacency matrix for road topology
│   ├── A_proximity.npy   # k-NN graph based on spatial distance
│   └── A_correlation.npy # Graph constructed from speed correlations
│
├── src/
│   ├── build_graphs.py   # Multi-view graph generation pipeline
│   ├── dataset.py        # PyTorch dataset loader for traffic data
│   ├── model_mvgnn.py    # Multi-view GNN model architecture
│   ├── train.py          # Model training pipeline
│   └── evaluate.py       # Evaluation metrics and visualizations
│
├── notebooks/
│   └── exploration.ipynb # Data exploration and experimental analysis
│
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.x |
| **ML Framework** | PyTorch |
| **Graph Learning** | PyTorch Geometric, NetworkX |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Geospatial (Optional)** | OSMnx, Folium, GeoPandas |
| **Visualization** | Matplotlib, Plotly, Seaborn |
| **UI (Optional)** | Streamlit |

---

## 🚀 How It Works

### Pipeline Overview

1. **Load Road Network + Traffic History**
   - Import road network topology (adjacency matrix or OSM data)
   - Load historical traffic speed/volume data

2. **Build Multi-View Graphs**
   - **Topology View:** Use road network connections
   - **Proximity View:** Compute k-nearest neighbors based on geographic distance
   - **Correlation View:** Calculate traffic pattern correlations between locations

3. **Neighbour Selection**
   - For each node in each graph view, select top-k most relevant neighbours
   - Reduces graph density while preserving important connections

4. **Train Multi-View GNN**
   - Learn spatial features through graph convolutions
   - Capture temporal dependencies with recurrent layers
   - Fuse multiple views for robust predictions

5. **Predict Traffic Speeds**
   - Forecast traffic speeds for next time intervals (e.g., 5–30 minutes ahead)
   - Generate predictions for all nodes in the network

6. **Visualize Results**
   - Compare predicted vs. actual traffic patterns
   - Generate heatmaps, time-series plots, and error distributions

### Algorithm Highlights

```
Input: Historical traffic data X_t, Road network G
Output: Predicted traffic speeds X_{t+h}

1. Construct multi-view graphs:
   - G_physical from road topology
   - G_proximity from spatial k-NN
   - G_correlation from traffic patterns

2. For each view v in {physical, proximity, correlation}:
   - Apply graph convolution: H_v = GCN(X, G_v)
   
3. Fuse multi-view features:
   - H_fused = Attention(H_physical, H_proximity, H_correlation)
   
4. Apply temporal layer:
   - Y = GRU(H_fused)
   
5. Generate predictions:
   - X_{t+h} = MLP(Y)
```

---

## 🏁 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Transport_Systems
cd Transport_Systems

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Generate multi-view graphs from raw data
python src/build_graphs.py

# 2. Train the TRAF-GNN model
python src/train.py --epochs 100 --lr 0.001 --k_neighbors 10

# 3. Evaluate and visualize results
python src/evaluate.py --checkpoint best_model.pth
```

### Configuration

Key parameters can be adjusted in the training script or via command-line arguments:

- `--k_neighbors`: Number of top neighbours to select (default: 10)
- `--hidden_dim`: Hidden dimension size for GNN layers (default: 64)
- `--num_layers`: Number of GNN layers (default: 3)
- `--sequence_length`: Input time window length (default: 12)
- `--prediction_horizon`: Forecast horizon (default: 3)

---

## 📈 Example Output

Once trained, TRAF-GNN produces:

### 1. **Predicted vs. Actual Speed Curves**
Time-series comparison showing model accuracy across different time periods.

### 2. **Congestion Heatmap**
Spatial visualization of predicted traffic intensity on the road network.

### 3. **Performance Metrics**
- **MAE (Mean Absolute Error):** Average prediction error in speed units
- **RMSE (Root Mean Square Error):** Overall prediction accuracy
- **MAPE (Mean Absolute Percentage Error):** Relative error percentage

> 📸 *Screenshots and visualizations will be added as the model is developed and tested.*

---

## 🧪 Use Cases

### Urban Traffic Management
- Predict congestion hotspots before they occur
- Enable proactive traffic signal optimization
- Support dynamic routing recommendations

### Smart City Applications
- Real-time congestion alerts for mobile apps
- Integration with adaptive traffic control systems
- Data-driven infrastructure planning

### Transportation Research
- Benchmark new graph learning algorithms
- Study spatio-temporal dynamics in urban networks
- Validate traffic flow theories with ML models

### Routing Algorithms
- Improve navigation systems with traffic forecasts
- Optimize fleet management and logistics
- Support emergency vehicle routing

---

## 📋 Roadmap

### ✅ Completed
- ✔️ Architecture designed
- ✔️ Multi-view graph creation strategy planned
- ✔️ Project structure established

### 🔜 In Progress
- 🚧 Multi-view graph generation implementation
- 🚧 TRAF-GNN model architecture coding
- 🚧 Training and evaluation pipeline

### 🎯 Future Enhancements
- [ ] Support for additional datasets (METR-LA, PeMS-BAY, etc.)
- [ ] Attention-based view fusion mechanisms
- [ ] Real-time inference API
- [ ] Streamlit dashboard for interactive predictions
- [ ] Transfer learning across cities
- [ ] Integration with OpenStreetMap for custom networks

---

## 🤝 Contributing

Contributions are welcome! This project is actively evolving. Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- Adding support for new traffic datasets
- Implementing alternative GNN architectures
- Improving visualization tools
- Optimizing training efficiency
- Documentation improvements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

If you use this code in your research, please consider citing:

```bibtex
@misc{trafgnn2025,
  title={TRAF-GNN: Multi-View Graph Learning for Traffic Forecasting},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-username/Transport_Systems}
}
```

### Related Work
- **DCRNN:** Diffusion Convolutional Recurrent Neural Network (Li et al., 2018)
- **Graph WaveNet:** Graph Neural Networks for Traffic Forecasting (Wu et al., 2019)
- **STGCN:** Spatio-Temporal Graph Convolutional Networks (Yu et al., 2018)
- **MTGNN:** Multi-Graph Neural Network for Traffic Forecasting (Wu et al., 2020)

---

## 📧 Contact

For questions, suggestions, or collaborations:

- **Email:** your.email@example.com
- **GitHub:** [@your-username](https://github.com/your-username)
- **LinkedIn:** [Your Name](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent GNN library
- Transportation research community for open datasets
- Contributors and collaborators

---

<div align="center">
  <strong>⭐ If you find this project useful, please consider giving it a star! ⭐</strong>
  <br><br>
  Made with ❤️ for smarter, more efficient urban transportation
</div>
