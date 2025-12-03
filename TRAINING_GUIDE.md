# TRAF-GNN Training Guide (Google Colab)

Complete guide to train your traffic forecasting model on free GPU.

---

## Quick Start (Copy-Paste Commands)

```bash
# 1. Clone & setup
!git clone https://github.com/V4RSH1TH-R3DDY/Transport_Systems.git
%cd Transport_Systems
!pip install torch torchvision tqdm matplotlib

# 2. Data pipeline
!python src/download_data.py --dataset metr-la
!python src/preprocessing.py
!python src/demo_graphs.py

# 3. Train (create train_colab.py first - see below)
!python train_colab.py
```

---

## Step 1: Setup (5 min)

**Enable GPU:**
- Runtime → Change runtime type → T4 GPU → Save

**Verify GPU:**
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## Step 2: Install Dependencies (5 min)

```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas scipy matplotlib seaborn tqdm h5py networkx
```

---

## Step 3: Clone Repository (2 min)

```python
!git clone https://github.com/V4RSH1TH-R3DDY/Transport_Systems.git
%cd Transport_Systems
!ls -la
```

---

## Step 4: Download Data (3 min)

```python
!python src/download_data.py --dataset metr-la
!ls -lh data/raw/
```

---

## Step 5: Preprocess Data (5 min)

```python
!python src/preprocessing.py \
    --dataset metr-la \
    --seq_length 12 \
    --pred_horizon 3 \
    --train_ratio 0.7 \
    --val_ratio 0.1

!ls -lh data/processed/
```

---

## Step 6: Build Graphs (2 min)

```python
# Use demo graphs (207 nodes) for faster training
!python src/demo_graphs.py

# OR use real graphs (4,106 nodes) - 20x slower
# !python src/build_real_graphs.py

!ls -lh graphs/
```

---

## Step 7: Create Training Script

Create `train_colab.py`:

```python
"""Training script for TRAF-GNN (Google Colab)"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append('/content/Transport_Systems/src')
from model_mvgnn import create_model
from dataset import create_dataloaders

# Configuration
CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'hidden_dim': 64,
    'num_gnn_layers': 2,
    'num_temporal_layers': 2,
    'dropout': 0.3,
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y, graphs) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        graphs = {k: v.to(device) for k, v in graphs.items()}
        
        optimizer.zero_grad()
        output = model(x, graphs)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y, graphs in val_loader:
            x = x.to(device)
            y = y.to(device)
            graphs = {k: v.to(device) for k, v in graphs.items()}
            
            output = model(x, graphs)
            loss = criterion(output, y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def calculate_metrics(model, test_loader, scaler, device):
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for x, y, graphs in test_loader:
            x = x.to(device)
            graphs = {k: v.to(device) for k, v in graphs.items()}
            
            output = model(x, graphs)
            predictions.append(output.cpu().numpy())
            targets.append(y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Denormalize
    predictions = scaler['mean'] + predictions * scaler['std']
    targets = scaler['mean'] + targets * scaler['std']
    
    # Metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-5))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def main():
    print("="*70)
    print("🚦 TRAF-GNN Training")
    print("="*70)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=CONFIG['batch_size'],
        use_demo_graphs=True
    )
    
    # Create model
    x, y, graphs = next(iter(train_loader))
    num_nodes = x.shape[2]
    
    model = create_model(num_nodes, config={
        'hidden_dim': CONFIG['hidden_dim'],
        'num_gnn_layers': CONFIG['num_gnn_layers'],
        'num_temporal_layers': CONFIG['num_temporal_layers'],
        'dropout': CONFIG['dropout'],
    })
    model = model.to(CONFIG['device'])
    
    print(f"✓ Model: {model.count_parameters():,} parameters")
    print(f"✓ Device: {CONFIG['device']}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    Path('checkpoints').mkdir(exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss = validate(model, val_loader, criterion, CONFIG['device'])
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_model.pth')
            print("  ✓ Saved best model")
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with open('data/processed/metr-la_stats.json', 'r') as f:
        scaler = json.load(f)
    
    metrics = calculate_metrics(model, test_loader, scaler, CONFIG['device'])
    
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS")
    print("="*70)
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("="*70)
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == '__main__':
    main()
```

---

## Step 8: Train! (2-3 hours)

```python
!python train_colab.py
```

**Expected:**
- Each epoch: 1-2 minutes
- Total: 2-3 hours for 100 epochs
- Early stopping may finish sooner

---

## Step 9: Download Results

```python
from google.colab import files

# Download model
files.download('checkpoints/best_model.pth')

# Download training curve
files.download('training_curve.png')
```

---

## Performance Targets

**Goal: Beat DCRNN (MAE 2.77)**

- **Good:** MAE 2.5-3.0, RMSE 5-6
- **Great:** MAE <2.7, RMSE <5.4
- **SOTA:** MAE <2.7, RMSE <5.2

---

## Troubleshooting

**GPU Out of Memory:**
```python
# Reduce batch size
CONFIG['batch_size'] = 16  # or 8
```

**Slow Training:**
```python
# Verify GPU
print(next(model.parameters()).device)  # Should be cuda:0
!nvidia-smi
```

**Session Timeout:**
- Keep tab open
- Check progress every 30 min
- Use Colab Pro for longer sessions

---

## Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoint
!cp checkpoints/best_model.pth '/content/drive/My Drive/TRAF-GNN/'
```

---

## Next Steps After Training

1. Compare results to baselines (DCRNN, STGCN)
2. Tune hyperparameters if needed
3. Try real 4,106-node graphs
4. Document findings
5. Share results!

---

**Total Time:** ~4-6 hours  
**GPU Required:** Yes (free T4 is fine)  
**Estimated MAE:** 2.5-3.0 (vs DCRNN 2.77)
