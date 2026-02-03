"""Data loading utilities for H3 cell-based species occurrence data."""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import h3


class H3DataLoader:
    """Load and prepare H3 cell-based species occurrence data for model training."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the parquet file containing H3 cell data
        """
        self.data_path = Path(data_path)
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.week_columns: List[str] = []
        self.env_columns: List[str] = []
        
    def load_data(self) -> gpd.GeoDataFrame:
        """Load the H3 cell data from parquet file."""
        self.gdf = gpd.read_parquet(self.data_path)
        
        # Identify week columns (week_1 through week_48)
        self.week_columns = [col for col in self.gdf.columns if col.startswith('week_')]
        
        # Identify environmental feature columns (all except h3_index, geometry, and week columns)
        self.env_columns = [
            col for col in self.gdf.columns 
            if col not in self.week_columns and col not in ['h3_index', 'geometry']
        ]
        
        return self.gdf
    
    def get_h3_cells(self) -> np.ndarray:
        """Extract H3 cell indices."""
        if self.gdf is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.gdf['h3_index'].values
    
    def h3_to_latlon(self, h3_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert H3 cell indices to latitude and longitude coordinates.
        
        Args:
            h3_cells: Array of H3 cell indices
            
        Returns:
            Tuple of (latitudes, longitudes) arrays in degrees
        """
        lats = []
        lons = []
        
        for cell in h3_cells:
            lat, lon = h3.cell_to_latlng(cell)
            lats.append(lat)
            lons.append(lon)
        
        return np.array(lats), np.array(lons)
    
    def get_environmental_features(self) -> pd.DataFrame:
        """
        Extract environmental/geographic features for each H3 cell.
        
        Returns:
            DataFrame with environmental features
        """
        if self.gdf is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.gdf[self.env_columns]
    
    def get_species_for_week(self, week: int) -> pd.Series:
        """
        Extract species lists for a specific week.
        
        Args:
            week: Week number (1-48)
            
        Returns:
            Series containing lists of GBIF taxonKeys for each H3 cell
        """
        if self.gdf is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        week_col = f'week_{week}'
        if week_col not in self.week_columns:
            raise ValueError(f"Week {week} not found in data. Valid weeks: 1-48")
        
        return self.gdf[week_col]
    
    def flatten_to_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]], pd.DataFrame]:
        """
        Flatten the data from H3 cell × weeks format to individual samples.
        
        Each row becomes 48 samples (one per week).
        
        Returns:
            lats: Array of latitudes (repeated for each week)
            lons: Array of longitudes (repeated for each week)
            weeks: Array of week numbers (1-48)
            species_lists: List of species lists (GBIF taxonKeys) for each sample
            env_features: Environmental features for each sample (repeated for each week)
        """
        if self.gdf is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        lats = []
        lons = []
        weeks = []
        species_lists = []
        env_features_list = []
        
        env_data = self.get_environmental_features()
        h3_cells = self.get_h3_cells()
        
        # Convert all H3 cells to lat/lon once
        cell_lats, cell_lons = self.h3_to_latlon(h3_cells)
        
        for idx, row in self.gdf.iterrows():
            lat = cell_lats[idx]
            lon = cell_lons[idx]
            
            for week_num in range(1, 49):  # weeks 1-48
                week_col = f'week_{week_num}'
                species_list = row[week_col]
                
                lats.append(lat)
                lons.append(lon)
                weeks.append(week_num)
                species_lists.append(species_list)
                env_features_list.append(env_data.iloc[idx].values)
        
        lats = np.array(lats)
        lons = np.array(lons)
        weeks = np.array(weeks)
        env_features_df = pd.DataFrame(env_features_list, columns=self.env_columns)
        
        return lats, lons, weeks, species_lists, env_features_df
    
    def get_data_info(self) -> Dict:
        """Get information about the loaded data."""
        if self.gdf is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'n_h3_cells': len(self.gdf),
            'n_weeks': len(self.week_columns),
            'n_environmental_features': len(self.env_columns),
            'environmental_feature_names': self.env_columns,
            'week_columns': self.week_columns
        }