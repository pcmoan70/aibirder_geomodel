"""Data preprocessing functions for H3 cell-based species occurrence data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, List, Set


class H3DataPreprocessor:
    """Preprocess H3 cell and species occurrence data for multi-task learning."""
    
    def __init__(self):
        self.env_scaler = StandardScaler()
        self.species_vocab: Set[int] = set()  # All unique GBIF taxonKeys
        self.species_to_idx: Dict[int, int] = {}
        self.idx_to_species: Dict[int, int] = {}
        self.env_feature_names: Optional[List[str]] = None
    
    def sinusoidal_encode_coordinates(
        self,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> np.ndarray:
        """
        Apply sinusoidal encoding to latitude and longitude coordinates.
        
        This encoding is suitable for neural networks as it:
        - Preserves continuity (e.g., -180° and 180° longitude map to same point)
        - Represents spherical nature of Earth
        
        Args:
            lats: Array of latitudes in degrees (-90 to 90)
            lons: Array of longitudes in degrees (-180 to 180)
            
        Returns:
            Array of shape (n_samples, 4) with [sin(lat), cos(lat), sin(lon), cos(lon)]
        """
        # Convert to radians
        lat_rad = np.deg2rad(lats)
        lon_rad = np.deg2rad(lons)
        
        # Apply sinusoidal encoding
        encoded = np.column_stack([
            np.sin(lat_rad),
            np.cos(lat_rad),
            np.sin(lon_rad),
            np.cos(lon_rad)
        ])
        
        return encoded.astype(np.float32)
    
    def normalize_environmental_features(
        self, 
        env_features: pd.DataFrame, 
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize environmental features.
        
        Args:
            env_features: DataFrame with environmental features
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized features as numpy array
        """
        if fit:
            self.env_feature_names = list(env_features.columns)
            return self.env_scaler.fit_transform(env_features)
        return self.env_scaler.transform(env_features)
    
    def build_species_vocabulary(
        self, 
        species_lists: List[List[int]]
    ) -> None:
        """
        Build vocabulary of all unique species (GBIF taxonKeys).
        
        Args:
            species_lists: List of species lists, where each list contains GBIF taxonKeys
        """
        all_species = set()
        for species_list in species_lists:
            if species_list.size > 0:  # Handle empty lists
                all_species.update(species_list)
        
        self.species_vocab = all_species
        self.species_to_idx = {species: idx for idx, species in enumerate(sorted(all_species))}
        self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
    
    def encode_species_multilabel(
        self, 
        species_lists: List[List[int]]
    ) -> np.ndarray:
        """
        Convert species lists to multi-label binary format.
        
        Args:
            species_lists: List of species lists (GBIF taxonKeys)
            
        Returns:
            Binary matrix of shape (n_samples, n_species)
        """
        if not self.species_vocab:
            self.build_species_vocabulary(species_lists)
            if not self.species_vocab:
                raise ValueError("Species vocabulary could not be built.")
        
        n_samples = len(species_lists)
        n_species = len(self.species_vocab)
        
        binary_matrix = np.zeros((n_samples, n_species), dtype=np.float32)
        
        for i, species_list in enumerate(species_lists):
            for species_id in species_list:
                if species_id in self.species_to_idx:
                    binary_matrix[i, self.species_to_idx[species_id]] = 1.0
        
        return binary_matrix
    
    def prepare_training_data(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        weeks: np.ndarray,
        species_lists: List[List[int]],
        env_features: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare all data for training with multi-task learning setup.
        
        Args:
            lats: Array of latitudes in degrees
            lons: Array of longitudes in degrees
            weeks: Array of week numbers (1-48)
            species_lists: List of species lists (GBIF taxonKeys)
            env_features: DataFrame with environmental features
            fit: Whether to fit encoders/scalers
            
        Returns:
            inputs: Dict with 'coordinates' (sinusoidal encoded) and 'week' arrays
            targets: Dict with 'species' (binary matrix) and 'env_features' arrays
        """
        # Sinusoidal encode coordinates
        encoded_coords = self.sinusoidal_encode_coordinates(lats, lons)
        
        # Normalize environmental features
        normalized_env = self.normalize_environmental_features(env_features, fit=fit)
        
        # Build species vocabulary and encode to multi-label format
        if fit:
            self.build_species_vocabulary(species_lists)
        species_binary = self.encode_species_multilabel(species_lists)
        
        # Prepare inputs
        inputs = {
            'coordinates': encoded_coords,  # Shape: (n_samples, 4)
            'week': weeks - 1  # Convert to 0-indexed (0-47)
        }
        
        # Prepare targets
        targets = {
            'species': species_binary,  # Multi-label binary
            'env_features': normalized_env  # Auxiliary target for training
        }
        
        return inputs, targets
    
    def split_data(
        self,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        split_by_location: bool = True
    ) -> Tuple[Dict[str, np.ndarray], ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            inputs: Dictionary of input arrays
            targets: Dictionary of target arrays
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed
            split_by_location: If True, group by unique locations to avoid data leakage
            
        Returns:
            train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets
        """
        n_samples = len(inputs['coordinates'])
        indices = np.arange(n_samples)
        
        if split_by_location:
            # Group by unique coordinate pairs to avoid data leakage
            # Create a hash of coordinate pairs to identify unique locations
            coord_tuples = [tuple(coord) for coord in inputs['coordinates']]
            unique_coords_map = {}
            location_indices = []
            
            for i, coord in enumerate(coord_tuples):
                if coord not in unique_coords_map:
                    unique_coords_map[coord] = len(unique_coords_map)
                location_indices.append(unique_coords_map[coord])
            
            location_indices = np.array(location_indices)
            unique_locations = np.unique(location_indices)
            
            # Split locations into train/test
            locs_train, locs_test = train_test_split(
                unique_locations, test_size=test_size, random_state=random_state
            )
            
            # Further split train locations into train/val
            val_ratio = val_size / (1 - test_size)
            locs_train, locs_val = train_test_split(
                locs_train, test_size=val_ratio, random_state=random_state
            )
            
            # Get indices for each split
            train_mask = np.isin(location_indices, locs_train)
            val_mask = np.isin(location_indices, locs_val)
            test_mask = np.isin(location_indices, locs_test)
        else:
            # Random split
            idx_temp, idx_test = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )
            val_ratio = val_size / (1 - test_size)
            idx_train, idx_val = train_test_split(
                idx_temp, test_size=val_ratio, random_state=random_state
            )
            
            train_mask = np.isin(indices, idx_train)
            val_mask = np.isin(indices, idx_val)
            test_mask = np.isin(indices, idx_test)
        
        # Split inputs and targets
        train_inputs = {k: v[train_mask] for k, v in inputs.items()}
        val_inputs = {k: v[val_mask] for k, v in inputs.items()}
        test_inputs = {k: v[test_mask] for k, v in inputs.items()}
        
        train_targets = {k: v[train_mask] for k, v in targets.items()}
        val_targets = {k: v[val_mask] for k, v in targets.items()}
        test_targets = {k: v[test_mask] for k, v in targets.items()}
        
        return (train_inputs, val_inputs, test_inputs, 
                train_targets, val_targets, test_targets)
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about preprocessing steps."""
        return {
            'n_species': len(self.species_vocab),
            'n_env_features': len(self.env_feature_names) if self.env_feature_names else 0,
            'env_feature_names': self.env_feature_names,
            'env_scaler_mean': self.env_scaler.mean_.tolist() if hasattr(self.env_scaler, 'mean_') else None,
            'env_scaler_scale': self.env_scaler.scale_.tolist() if hasattr(self.env_scaler, 'scale_') else None,
            'species_vocab_size': len(self.species_vocab),
            'sample_species_ids': list(self.species_vocab)[:10],  # First 10 for reference
            'coordinate_encoding': 'sinusoidal',
            'coordinate_features': 4  # sin(lat), cos(lat), sin(lon), cos(lon)
        }