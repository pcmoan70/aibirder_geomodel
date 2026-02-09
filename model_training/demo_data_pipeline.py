"""
Demonstration script for data loading and preprocessing pipeline.

This script shows how to:
1. Load H3 cell-based species occurrence data
2. Flatten to individual samples (cell × week)
3. Preprocess data for neural network training
4. Display example inputs and targets
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model_training.data.loader import H3DataLoader
from model_training.data.preprocessing import H3DataPreprocessor


def print_separator(title: str = ""):
    """Print a separator line with optional title."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'-'*80}\n")


def demonstrate_data_loading(data_path: str):
    """Demonstrate the complete data loading and preprocessing pipeline."""
    
    print_separator("DATA LOADING DEMONSTRATION")
    
    # Initialize loader
    print("1. Initializing H3DataLoader...")
    loader = H3DataLoader(data_path)
    
    # Load data
    print("2. Loading data from parquet file...")
    loader.load_data()
    
    # Display data info
    print_separator("Dataset Information")
    data_info = loader.get_data_info()
    print(f"Number of H3 cells: {data_info['n_h3_cells']}")
    print(f"Number of weeks: {data_info['n_weeks']}")
    print(f"Number of environmental features: {data_info['n_environmental_features']}")
    print(f"\nEnvironmental features ({len(data_info['environmental_feature_names'])}):")
    for i, feature in enumerate(data_info['environmental_feature_names'][:10], 1):
        print(f"  {i}. {feature}")
    if len(data_info['environmental_feature_names']) > 10:
        print(f"  ... and {len(data_info['environmental_feature_names']) - 10} more")
    
    # Flatten to samples
    print_separator("Flattening Data to Samples")
    print("3. Converting H3 cells × weeks to individual samples...")
    lats, lons, weeks, species_lists, env_features = loader.flatten_to_samples()
    
    print("\nFlattened data shapes:")
    print(f"  Latitudes: {lats.shape}")
    print(f"  Longitudes: {lons.shape}")
    print(f"  Weeks: {weeks.shape}")
    print(f"  Species lists: {len(species_lists)} samples")
    print(f"  Environmental features: {env_features.shape}")
    
    print("\nExample samples (first 3):")
    for i in range(min(3, len(lats))):
        print(f"\nSample {i+1}:")
        print(f"  Location: ({lats[i]:.4f}°, {lons[i]:.4f}°)")
        print(f"  Week: {weeks[i]}")
        print(f"  Number of species: {len(species_lists[i])}")
        if len(species_lists[i]) > 0:
            print(f"  First 5 species (GBIF taxonKeys): {species_lists[i][:5]}")
    
    # Preprocessing
    print_separator("DATA PREPROCESSING")
    print("4. Initializing H3DataPreprocessor...")
    preprocessor = H3DataPreprocessor()
    
    print("5. Preparing training data...")
    print("   - Encoding coordinates with sinusoidal encoding")
    print("   - Encoding weeks with sinusoidal encoding (cyclical)")
    print("   - Normalizing environmental features")
    print("   - Building species vocabulary")
    print("   - Converting species lists to multi-label binary format")
    
    inputs, targets = preprocessor.prepare_training_data(
        lats=lats,
        lons=lons,
        weeks=weeks,
        species_lists=species_lists,
        env_features=env_features,
        fit=True
    )
    
    # Display preprocessing info
    print_separator("Preprocessing Information")
    prep_info = preprocessor.get_preprocessing_info()
    print(f"Total unique species (vocabulary size): {prep_info['n_species']}")
    print(f"Number of environmental features: {prep_info['n_env_features']}")
    print(f"Coordinate encoding: {prep_info['coordinate_encoding']}")
    print(f"Coordinate features: {prep_info['coordinate_features']}")
    
    print("\nSample species IDs (first 10 GBIF taxonKeys):")
    for taxon_key in prep_info['sample_species_ids'][:10]:
        print(f"  - {taxon_key}")
    
    # Display input shapes and examples
    print_separator("MODEL INPUTS")
    print("Shape of inputs that will be fed to the neural network:\n")
    print("Coordinates (sinusoidal encoded):")
    print(f"  Shape: {inputs['coordinates'].shape}")
    print("  Description: 4 features per sample [sin(lat), cos(lat), sin(lon), cos(lon)]")
    print(f"  Data type: {inputs['coordinates'].dtype}")
    
    print("\nWeek (sinusoidal encoded):")
    print(f"  Shape: {inputs['week'].shape}")
    print("  Description: 2 features per sample [sin(week), cos(week)] for cyclical encoding")
    print(f"  Data type: {inputs['week'].dtype}")
    
    print("\nExample input samples (first 3):")
    for i in range(min(3, len(inputs['coordinates']))):
        print(f"\nSample {i+1}:")
        print(f"  Coordinates (encoded): {inputs['coordinates'][i]}")
        print(f"  Week (encoded): {inputs['week'][i]}")
    
    # Display target shapes and examples
    print_separator("MODEL TARGETS")
    print("Shape of targets used during training:\n")
    print("Species (PRIMARY TARGET - multi-label binary):")
    print(f"  Shape: {targets['species'].shape}")
    print("  Description: Binary vector indicating presence/absence of each species")
    print(f"  Data type: {targets['species'].dtype}")
    print(f"  Sparsity: {np.mean(targets['species'] == 0) * 100:.2f}% zeros (sparse)")
    
    print("\nEnvironmental features (AUXILIARY TARGET):")
    print(f"  Shape: {targets['env_features'].shape}")
    print("  Description: Normalized environmental features for regularization")
    print(f"  Data type: {targets['env_features'].dtype}")
    
    print("\nExample target samples (first 3):")
    for i in range(min(3, len(targets['species']))):
        n_species_present = int(targets['species'][i].sum())
        species_indices = np.where(targets['species'][i] == 1)[0][:5]  # First 5 present species
        
        print(f"\nSample {i+1}:")
        print(f"  Species present: {n_species_present} out of {targets['species'].shape[1]}")
        print(f"  First 5 present species (indices): {species_indices.tolist()}")
        print(f"  Environmental features (first 5): {targets['env_features'][i][:5]}")
    
    # Splitting data
    print_separator("DATA SPLITTING")
    print("6. Splitting data into train/val/test sets...")
    print("   Using location-based splitting to avoid data leakage")
    
    train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets = \
        preprocessor.split_data(
            inputs=inputs,
            targets=targets,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            split_by_location=True
        )
    
    print("\nDataset splits:")
    print(f"  Training set:   {len(train_inputs['coordinates']):,} samples")
    print(f"  Validation set: {len(val_inputs['coordinates']):,} samples")
    print(f"  Test set:       {len(test_inputs['coordinates']):,} samples")
    print(f"  Total:          {len(inputs['coordinates']):,} samples")
    
    train_pct = len(train_inputs['coordinates']) / len(inputs['coordinates']) * 100
    val_pct = len(val_inputs['coordinates']) / len(inputs['coordinates']) * 100
    test_pct = len(test_inputs['coordinates']) / len(inputs['coordinates']) * 100
    
    print("\nSplit percentages:")
    print(f"  Training:   {train_pct:.1f}%")
    print(f"  Validation: {val_pct:.1f}%")
    print(f"  Test:       {test_pct:.1f}%")
    
    # Summary
    print_separator("SUMMARY")
    print("Data pipeline successfully demonstrated!")
    print("\nKey points:")
    print("  ✓ Data loaded from parquet using GeoPandas")
    print("  ✓ H3 cells converted to lat/lon coordinates")
    print("  ✓ Coordinates encoded with sinusoidal encoding for neural networks")
    print("  ✓ Species lists converted to multi-label binary format")
    print("  ✓ Environmental features normalized (used as auxiliary target)")
    print("  ✓ Data split by location to prevent leakage")
    print("\nThe data is now ready to be fed into a multi-task neural network!")
    print("  - Input: (coordinates, week)")
    print("  - Primary output: species presence/absence")
    print("  - Auxiliary output: environmental features (training only)")
    
    print_separator()
    
    return {
        'loader': loader,
        'preprocessor': preprocessor,
        'inputs': inputs,
        'targets': targets,
        'splits': (train_inputs, val_inputs, test_inputs, 
                   train_targets, val_targets, test_targets)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate the data loading and preprocessing pipeline"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./outputs/global_350km_ee_gbif.parquet",
        help="Path to the parquet file containing H3 cell data",
        required=False
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Run demonstration
    try:
        results = demonstrate_data_loading(str(data_path))
        print("\n✓ Demonstration completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
