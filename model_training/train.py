"""
Training script for BirdNET Geomodel.

This script implements the complete training pipeline:
1. Load and preprocess data
2. Initialize model, optimizer, and loss function
3. Train with validation
4. Save checkpoints and metrics
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model_training.data.loader import H3DataLoader
from model_training.data.preprocessing import H3DataPreprocessor
from model_training.data.dataset import create_dataloaders, get_class_weights
from model_training.model.model import create_model
from model_training.model.loss import MultiTaskLoss


class Trainer:
    """Trainer class for BirdNET Geomodel."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: Path,
        log_interval: int = 10
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on (CPU or CUDA)
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging frequency (batches)
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_species_loss': [],
            'train_env_loss': [],
            'val_loss': [],
            'val_species_loss': [],
            'val_env_loss': []
        }
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_species_loss = 0.0
        total_env_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            coordinates = inputs['coordinates'].to(self.device)
            week = inputs['week'].to(self.device)
            species_targets = targets['species'].to(self.device)
            env_targets = targets['env_features'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(coordinates, week, return_env=True)
            
            # Compute loss
            losses = self.criterion(
                outputs,
                {'species': species_targets, 'env_features': env_targets}
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total'].item()
            total_species_loss += losses['species'].item()
            total_env_loss += losses['env'].item()
            num_batches += 1
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'species': f"{losses['species'].item():.4f}",
                    'env': f"{losses['env'].item():.4f}"
                })
        
        return {
            'loss': total_loss / num_batches,
            'species_loss': total_species_loss / num_batches,
            'env_loss': total_env_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with average losses
        """
        self.model.eval()
        
        total_loss = 0.0
        total_species_loss = 0.0
        total_env_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]  ')
        
        for inputs, targets in pbar:
            # Move to device
            coordinates = inputs['coordinates'].to(self.device)
            week = inputs['week'].to(self.device)
            species_targets = targets['species'].to(self.device)
            env_targets = targets['env_features'].to(self.device)
            
            # Forward pass
            outputs = self.model(coordinates, week, return_env=True)
            
            # Compute loss
            losses = self.criterion(
                outputs,
                {'species': species_targets, 'env_features': env_targets}
            )
            
            # Accumulate losses
            total_loss += losses['total'].item()
            total_species_loss += losses['species'].item()
            total_env_loss += losses['env'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'species_loss': total_species_loss / num_batches,
            'env_loss': total_env_loss / num_batches
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt'
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_every: int = 5
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset):,}")
        print(f"Validation samples: {len(val_loader.dataset):,}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Batches per epoch: {len(train_loader)}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_species_loss'].append(train_metrics['species_loss'])
            self.history['train_env_loss'].append(train_metrics['env_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_species_loss'].append(val_metrics['species_loss'])
            self.history['val_env_loss'].append(val_metrics['env_loss'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Species: {train_metrics['species_loss']:.4f}, "
                  f"Env: {train_metrics['env_loss']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Species: {val_metrics['species_loss']:.4f}, "
                  f"Env: {val_metrics['env_loss']:.4f}")
            
            # Check if best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        # Save final training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n✓ Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train BirdNET Geomodel')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='./outputs/global_350km_ee_gbif.parquet',
                       help='Path to parquet data file')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size configuration')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--species_weight', type=float, default=1.0,
                       help='Weight for species loss')
    parser.add_argument('--env_weight', type=float, default=0.5,
                       help='Weight for environmental loss')
    
    # Data split arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set proportion (of training data)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("  BirdNET Geomodel Training")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Data: {args.data_path}")
    print(f"  Model size: {args.model_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    
    # Load and preprocess data
    print("\n" + "="*80)
    print("  Loading and Preprocessing Data")
    print("="*80)
    
    print("\n1. Loading data...")
    loader = H3DataLoader(args.data_path)
    loader.load_data()
    
    print("2. Flattening to samples...")
    lats, lons, weeks, species_lists, env_features = loader.flatten_to_samples()
    
    print("3. Preprocessing...")
    preprocessor = H3DataPreprocessor()
    inputs, targets = preprocessor.prepare_training_data(
        lats=lats,
        lons=lons,
        weeks=weeks,
        species_lists=species_lists,
        env_features=env_features,
        fit=True
    )
    
    print("\nDataset info:")
    print(f"  Total samples: {len(inputs['coordinates']):,}")
    print(f"  Number of species: {preprocessor.get_preprocessing_info()['n_species']:,}")
    print(f"  Environmental features: {preprocessor.get_preprocessing_info()['n_env_features']}")
    
    print("\n4. Splitting data...")
    train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets = \
        preprocessor.split_data(
            inputs=inputs,
            targets=targets,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=42,
            split_by_location=True
        )
    
    print(f"  Training: {len(train_inputs['coordinates']):,} samples")
    print(f"  Validation: {len(val_inputs['coordinates']):,} samples")
    print(f"  Test: {len(test_inputs['coordinates']):,} samples")
    
    # Create data loaders
    print("\n5. Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # Compute class weights
    print("\n6. Computing class weights for species...")
    pos_weights = get_class_weights(train_targets['species'])
    # Disable pos_weights initially to debug
    # pos_weights = pos_weights.to(device)
    pos_weights = None
    if pos_weights is not None:
        print(f"  Weight range: {pos_weights.min():.2f} - {pos_weights.max():.2f}")
    
    # Create model
    print("\n" + "="*80)
    print("  Initializing Model")
    print("="*80)
    
    n_species = preprocessor.get_preprocessing_info()['n_species']
    n_env_features = preprocessor.get_preprocessing_info()['n_env_features']
    
    model = create_model(
        n_species=n_species,
        n_env_features=n_env_features,
        model_size=args.model_size
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {args.model_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Create loss function
    criterion = MultiTaskLoss(
        species_weight=args.species_weight,
        env_weight=args.env_weight,
        pos_weight=pos_weights
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create trainer
    checkpoint_dir = Path(args.checkpoint_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Train
    print("\n" + "="*80)
    print("  Training")
    print("="*80)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_every=args.save_every
    )
    
    print("\n✓ Training completed successfully!")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
