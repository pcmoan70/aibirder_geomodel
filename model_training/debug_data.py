"""Debug script to check for NaN values and data ranges."""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from model_training.data.loader import H3DataLoader
from model_training.data.preprocessing import H3DataPreprocessor

def check_data():
    """Check for NaN values and data ranges."""
    
    print("Loading data...")
    loader = H3DataLoader("./outputs/global_350km_ee_gbif.parquet")
    loader.load_data()
    
    lats, lons, weeks, species_lists, env_features = loader.flatten_to_samples()
    
    print("\nRaw data checks:")
    print(f"  Lats - min: {lats.min():.4f}, max: {lats.max():.4f}, NaN: {np.isnan(lats).sum()}")
    print(f"  Lons - min: {lons.min():.4f}, max: {lons.max():.4f}, NaN: {np.isnan(lons).sum()}")
    print(f"  Weeks - min: {weeks.min()}, max: {weeks.max()}, NaN: {np.isnan(weeks).sum()}")
    print(f"  Env features - NaN: {env_features.isna().sum().sum()}")
    print(f"  Env features shape: {env_features.shape}")
    
    print("\nEnv features stats:")
    print(env_features.describe())
    
    print("\nPreprocessing...")
    preprocessor = H3DataPreprocessor()
    inputs, targets = preprocessor.prepare_training_data(
        lats=lats,
        lons=lons,
        weeks=weeks,
        species_lists=species_lists,
        env_features=env_features,
        fit=True
    )
    
    print("\nPreprocessed data checks:")
    print(f"  Coordinates - NaN: {np.isnan(inputs['coordinates']).sum()}")
    print(f"  Coordinates - min: {inputs['coordinates'].min():.4f}, max: {inputs['coordinates'].max():.4f}")
    print(f"  Week - NaN: {np.isnan(inputs['week']).sum()}")
    print(f"  Week - min: {inputs['week'].min():.4f}, max: {inputs['week'].max():.4f}")
    print(f"  Species - NaN: {np.isnan(targets['species']).sum()}")
    print(f"  Env features - NaN: {np.isnan(targets['env_features']).sum()}")
    print(f"  Env features - min: {targets['env_features'].min():.4f}, max: {targets['env_features'].max():.4f}")
    print(f"  Env features - mean: {targets['env_features'].mean():.4f}, std: {targets['env_features'].std():.4f}")

if __name__ == "__main__":
    check_data()
