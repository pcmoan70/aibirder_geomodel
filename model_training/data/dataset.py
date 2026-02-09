"""
PyTorch Dataset and DataLoader utilities for BirdNET Geomodel.

This module provides dataset classes and data loading utilities for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


class BirdSpeciesDataset(Dataset):
    """
    PyTorch Dataset for bird species occurrence prediction.
    
    Handles the prepared inputs and targets from the preprocessing pipeline.
    """
    
    def __init__(
        self,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ):
        """
        Initialize the dataset.
        
        Args:
            inputs: Dictionary with 'coordinates' and 'week' arrays
            targets: Dictionary with 'species' and 'env_features' arrays
        """
        self.coordinates = torch.from_numpy(inputs['coordinates']).float()
        self.week = torch.from_numpy(inputs['week']).float()
        self.species = torch.from_numpy(targets['species']).float()
        self.env_features = torch.from_numpy(targets['env_features']).float()
        
        # Verify all have same length
        assert len(self.coordinates) == len(self.week) == len(self.species) == len(self.env_features)
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.coordinates)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            inputs: Dictionary with 'coordinates' and 'week' tensors
            targets: Dictionary with 'species' and 'env_features' tensors
        """
        inputs = {
            'coordinates': self.coordinates[idx],
            'week': self.week[idx]
        }
        
        targets = {
            'species': self.species[idx],
            'env_features': self.env_features[idx]
        }
        
        return inputs, targets


def create_dataloaders(
    train_inputs: Dict[str, np.ndarray],
    train_targets: Dict[str, np.ndarray],
    val_inputs: Dict[str, np.ndarray],
    val_targets: Dict[str, np.ndarray],
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_inputs: Training inputs dictionary
        train_targets: Training targets dictionary
        val_inputs: Validation inputs dictionary
        val_targets: Validation targets dictionary
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = BirdSpeciesDataset(train_inputs, train_targets)
    val_dataset = BirdSpeciesDataset(val_inputs, val_targets)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for batch norm stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def get_class_weights(
    species_targets: np.ndarray,
    smoothing: float = 100.0,
    max_weight: float = 50.0
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced species.
    
    Args:
        species_targets: Binary species labels [n_samples, n_species]
        smoothing: Smoothing factor to prevent extreme weights
        max_weight: Maximum weight to cap extreme values
        
    Returns:
        Positive class weights tensor [n_species]
    """
    species_tensor = torch.from_numpy(species_targets).float()
    
    # Count positive and negative examples per species
    pos_counts = species_tensor.sum(dim=0)
    neg_counts = (1 - species_tensor).sum(dim=0)
    
    # Compute weights with smoothing
    pos_weights = (neg_counts + smoothing) / (pos_counts + smoothing)
    
    # Cap extreme weights to prevent training instability
    pos_weights = torch.clamp(pos_weights, max=max_weight)
    
    return pos_weights
