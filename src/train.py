"""
Training Script for TRAF-GNN
Complete training pipeline with checkpointing, early stopping, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from model_mvgnn import create_model
from dataset import create_dataloaders
from metrics import masked_mae, masked_rmse, masked_mape, calculate_metrics
from config import get_default_config, save_config, load_config, update_config, print_config
from utils import (set_seed, get_device, count_parameters, print_model_summary,
                   load_scaler, EarlyStopping, create_experiment_dir)


class Trainer:
    """Training manager for TRAF-GNN"""
    
    def __init__(self, config: dict):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Set seed for reproducibility
        set_seed(config['experiment']['seed'])
        
        # Get device
        self.device = get_device(config['experiment']['device'])
        
        # Create directories
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Data loaders
        print("\n📊 Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            use_demo_graphs=config['data']['use_demo_graphs']
        )
        
        # Get number of nodes from data
        x, y, graphs = next(iter(self.train_loader))
        self.num_nodes = x.shape[2]
        
        # Create model
        print(f"\n🏗️  Creating model...")
        self.model = create_model(self.num_nodes, config=config['model'])
        self.model = self.model.to(self.device)
        
        print_model_summary(self.model)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config['scheduler']['patience'],
            factor=config['scheduler']['factor'],
            min_lr=config['scheduler']['min_lr']
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            min_epochs=config['training']['min_epochs']
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        # Load scaler
        stats_path = Path(config['paths']['data_dir']) / f"{config['data']['dataset']}_stats.json"
        if stats_path.exists():
            self.scaler = load_scaler(str(stats_path))
        else:
            print(f"⚠️  Warning: Stats file not found at {stats_path}")
            self.scaler = None
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, (x, y, graphs) in enumerate(pbar):
            # Move to device
            x = x.to(self.device)
            y = y.to(self.device)
            graphs = {k: v.to(self.device) for k, v in graphs.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x, graphs)
            loss = self.criterion(output, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            for x, y, graphs in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                graphs = {k: v.to(self.device) for k, v in graphs.items()}
                
                output = self.model(x, graphs)
                loss = self.criterion(output, y)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def test(self) -> dict:
        """Test model and calculate metrics"""
        self.model.eval()
        predictions = []
        targets = []
        
        print("\n📊 Evaluating on test set...")
        with torch.no_grad():
            for x, y, graphs in tqdm(self.test_loader):
                x = x.to(self.device)
                graphs = {k: v.to(self.device) for k, v in graphs.items()}
                
                output = self.model(x, graphs)
                
                predictions.append(output.cpu().numpy())
                targets.append(y.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets, self.scaler)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def plot_losses(self, save_path: str = None):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved loss plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("🚦 STARTING TRAINING")
        print("=" * 70)
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.early_stopping(val_loss, epoch):
                break
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot losses
        plot_path = self.checkpoint_dir / 'training_curve.png'
        self.plot_losses(str(plot_path))
        
        # Load best model for testing
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        if best_model_path.exists():
            self.load_checkpoint(str(best_model_path))
        
        # Test
        test_metrics = self.test()
        
        print("\n" + "=" * 70)
        print("🎯 TEST RESULTS")
        print("=" * 70)
        print(f"MAE:  {test_metrics['MAE']:.4f}")
        print(f"RMSE: {test_metrics['RMSE']:.4f}")
        print(f"MAPE: {test_metrics['MAPE']:.2f}%")
        print("=" * 70)
        
        # Save test results
        results_path = self.checkpoint_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train TRAF-GNN model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate model')
    
    # Override config options
    parser.add_argument('--num-epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension size')
    parser.add_argument('--use-demo-graphs', action='store_true', help='Use demo graphs')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line args
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.hidden_dim:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.use_demo_graphs:
        config['data']['use_demo_graphs'] = True
    
    # Print config
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print_config(config)
    print("=" * 70)
    
    # Save config
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(checkpoint_dir / 'config.json'))
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Evaluate only
    if args.evaluate:
        if args.resume:
            metrics = trainer.test()
            print("\n" + "=" * 70)
            print("EVALUATION RESULTS")
            print("=" * 70)
            print(f"MAE:  {metrics['MAE']:.4f}")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            print("=" * 70)
        else:
            print("Error: --resume required for --evaluate")
        return
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
