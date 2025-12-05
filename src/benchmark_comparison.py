"""
Benchmark Comparison: TRAF-GNN vs Baseline Models
Compares TRAF-GNN against DCRNN, STGCN, and other baselines on METR-LA dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Published benchmark results on METR-LA dataset
# Sources: Original papers and Graph WaveNet (Wu et al., 2019)

BASELINE_RESULTS = {
    # Prediction Horizon: 15 min (3 steps), 30 min (6 steps), 60 min (12 steps)
    
    # Historical Average (HA)
    'HA': {
        '15min': {'MAE': 4.16, 'RMSE': 7.80, 'MAPE': 13.0},
        '30min': {'MAE': 4.16, 'RMSE': 7.80, 'MAPE': 13.0},
        '60min': {'MAE': 4.16, 'RMSE': 7.80, 'MAPE': 13.0},
    },
    
    # ARIMA
    'ARIMA': {
        '15min': {'MAE': 3.99, 'RMSE': 8.21, 'MAPE': 9.6},
        '30min': {'MAE': 5.15, 'RMSE': 10.45, 'MAPE': 12.7},
        '60min': {'MAE': 6.90, 'RMSE': 13.23, 'MAPE': 17.4},
    },
    
    # FC-LSTM (Fully Connected LSTM)
    'FC-LSTM': {
        '15min': {'MAE': 3.44, 'RMSE': 6.30, 'MAPE': 9.6},
        '30min': {'MAE': 3.77, 'RMSE': 7.23, 'MAPE': 10.9},
        '60min': {'MAE': 4.37, 'RMSE': 8.69, 'MAPE': 13.2},
    },
    
    # DCRNN (Diffusion Convolutional Recurrent Neural Network)
    # Li et al., ICLR 2018
    'DCRNN': {
        '15min': {'MAE': 2.77, 'RMSE': 5.38, 'MAPE': 7.3},
        '30min': {'MAE': 3.15, 'RMSE': 6.45, 'MAPE': 8.8},
        '60min': {'MAE': 3.60, 'RMSE': 7.60, 'MAPE': 10.5},
    },
    
    # STGCN (Spatio-Temporal Graph Convolutional Network)
    # Yu et al., IJCAI 2018
    'STGCN': {
        '15min': {'MAE': 2.88, 'RMSE': 5.74, 'MAPE': 7.6},
        '30min': {'MAE': 3.47, 'RMSE': 7.24, 'MAPE': 9.6},
        '60min': {'MAE': 4.59, 'RMSE': 9.40, 'MAPE': 12.7},
    },
    
    # Graph WaveNet
    # Wu et al., IJCAI 2019
    'Graph WaveNet': {
        '15min': {'MAE': 2.69, 'RMSE': 5.15, 'MAPE': 6.9},
        '30min': {'MAE': 3.07, 'RMSE': 6.22, 'MAPE': 8.4},
        '60min': {'MAE': 3.53, 'RMSE': 7.37, 'MAPE': 10.0},
    },
    
    # MTGNN (Multi-faceted Graph Neural Network)
    # Wu et al., KDD 2020
    'MTGNN': {
        '15min': {'MAE': 2.69, 'RMSE': 5.18, 'MAPE': 6.86},
        '30min': {'MAE': 3.05, 'RMSE': 6.17, 'MAPE': 8.19},
        '60min': {'MAE': 3.49, 'RMSE': 7.23, 'MAPE': 9.87},
    },
    
    # TRAF-GNN (Our model - 2-layer, hidden_dim=128, 50 epochs)
    'TRAF-GNN (Ours)': {
        '15min': {'MAE': 3.45, 'RMSE': 7.31, 'MAPE': 7.87},
        '30min': {'MAE': None, 'RMSE': None, 'MAPE': None},  # Need longer horizon training
        '60min': {'MAE': None, 'RMSE': None, 'MAPE': None},  # Need longer horizon training
    },
}


def create_comparison_table():
    """Create comparison table for all models"""
    print("="*80)
    print("BENCHMARK COMPARISON: METR-LA Dataset")
    print("Traffic Speed Forecasting Performance")
    print("="*80)
    
    horizons = ['15min', '30min', '60min']
    metrics = ['MAE', 'RMSE', 'MAPE']
    
    for horizon in horizons:
        print(f"\n{'─'*80}")
        print(f"Prediction Horizon: {horizon} ({horizon.replace('min', '')} minutes ahead)")
        print(f"{'─'*80}")
        
        # Create table
        table_data = []
        for model_name, results in BASELINE_RESULTS.items():
            row = [model_name]
            for metric in metrics:
                value = results[horizon][metric]
                if value is None:
                    row.append("TBD")
                else:
                    row.append(f"{value:.2f}")
            table_data.append(row)
        
        df = pd.DataFrame(table_data, columns=['Model', 'MAE ↓', 'RMSE ↓', 'MAPE(%) ↓'])
        print(df.to_string(index=False))
        
        # Find best performers
        print(f"\n🏆 Best Performance @ {horizon}:")
        for metric in metrics:
            values = [(name, results[horizon][metric]) 
                     for name, results in BASELINE_RESULTS.items() 
                     if results[horizon][metric] is not None]
            if values:
                best_model, best_value = min(values, key=lambda x: x[1])
                print(f"  {metric}: {best_model} ({best_value:.2f})")


def create_visualization():
    """Create visual comparison"""
    horizons = ['15min', '30min', '60min']
    metrics = ['MAE', 'RMSE', 'MAPE']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        models_to_plot = ['DCRNN', 'STGCN', 'Graph WaveNet', 'MTGNN']
        
        for model_name in models_to_plot:
            values = [BASELINE_RESULTS[model_name][h][metric] for h in horizons]
            ax.plot(horizons, values, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('Prediction Horizon', fontsize=12)
        ax.set_ylabel(f'{metric}', fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/baseline_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to graphs/baseline_comparison.png")
    plt.close()


def calculate_improvement_needed():
    """Calculate what performance TRAF-GNN needs to beat SOTA"""
    print("\n" + "="*80)
    print("TARGET PERFORMANCE FOR TRAF-GNN")
    print("="*80)
    
    horizons = ['15min', '30min', '60min']
    metrics = ['MAE', 'RMSE', 'MAPE']
    
    print("\nTo match/beat current SOTA (MTGNN/Graph WaveNet):")
    print()
    
    for horizon in horizons:
        print(f"📊 {horizon} ahead:")
        for metric in metrics:
            # Find best value
            values = [results[horizon][metric] 
                     for name, results in BASELINE_RESULTS.items() 
                     if name != 'TRAF-GNN (Ours)' and results[horizon][metric] is not None]
            best_value = min(values)
            target = best_value * 0.95  # 5% improvement
            
            print(f"  {metric:8} - Current best: {best_value:.2f}, "
                  f"Target (5% better): {target:.2f}")
        print()


def generate_latex_table():
    """Generate LaTeX table for paper"""
    print("\n" + "="*80)
    print("LATEX TABLE (for research paper)")
    print("="*80)
    print()
    
    latex = r"""\begin{table}[h]
\centering
\caption{Performance Comparison on METR-LA Dataset}
\begin{tabular}{l|ccc|ccc|ccc}
\hline
\multirow{2}{*}{Model} & \multicolumn{3}{c|}{15 min} & \multicolumn{3}{c|}{30 min} & \multicolumn{3}{c}{60 min} \\
& MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE \\
\hline
"""
    
    models = ['HA', 'ARIMA', 'FC-LSTM', 'DCRNN', 'STGCN', 'Graph WaveNet', 'MTGNN', 'TRAF-GNN (Ours)']
    
    for model in models:
        if model not in BASELINE_RESULTS:
            continue
        
        row = model
        for horizon in ['15min', '30min', '60min']:
            mae = BASELINE_RESULTS[model][horizon]['MAE']
            rmse = BASELINE_RESULTS[model][horizon]['RMSE']
            mape = BASELINE_RESULTS[model][horizon]['MAPE']
            
            if mae is None:
                row += f" & - & - & -"
            else:
                row += f" & {mae:.2f} & {rmse:.2f} & {mape:.2f}"
        
        row += r" \\"
        latex += row + "\n"
    
    latex += r"""\hline
\end{tabular}
\label{tab:benchmark}
\end{table}"""
    
    print(latex)


if __name__ == '__main__':
    print("\n🚦 TRAF-GNN Benchmark Comparison Tool\n")
    
    # Show comparison table
    create_comparison_table()
    
    # Calculate targets
    calculate_improvement_needed()
    
    # Create visualization
    create_visualization()
    
    # Generate LaTeX
    generate_latex_table()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
📊 Current State-of-the-Art (METR-LA):
   • Best overall: MTGNN and Graph WaveNet
   • 15-min MAE: ~2.69 (Graph WaveNet/MTGNN)
   • 60-min MAE: ~3.49 (MTGNN)

🎯 TRAF-GNN Goal:
   • Match or beat SOTA performance
   • Target: <2.69 MAE @ 15min, <3.49 MAE @ 60min
   • Multi-view learning should help capture diverse patterns

🔬 Key Advantages of TRAF-GNN:
   1. Explicit multi-view architecture (physical + proximity + correlation)
   2. Attention-based fusion learns view importance
   3. Efficient k-NN neighbor selection reduces complexity
   4. Combines strengths of DCRNN (diffusion) and STGCN (spectral)

📝 Next Steps:
   1. Complete Phase 4: Training Pipeline
   2. Train model on METR-LA dataset
   3. Evaluate on test set with these metrics
   4. Update TRAF-GNN results in benchmark table
   5. Analyze where multi-view learning helps most

Note: TRAF-GNN results pending training (Phase 4)
    """)
    print("="*80)
