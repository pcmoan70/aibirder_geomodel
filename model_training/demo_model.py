"""
Demonstration script for the BirdNET Geomodel architecture.

This script shows how to:
1. Initialize the multi-task model
2. Process sample data through the model
3. Display model architecture and parameter counts
4. Demonstrate forward pass with example inputs
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model_training.model.model import create_model
from model_training.model.loss import MultiTaskLoss, compute_pos_weights


def print_separator(title: str = ""):
    """Print a separator line with optional title."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'-'*80}\n")


def count_parameters(model: torch.nn.Module) -> dict:
    """Count trainable and total parameters in model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {'trainable': trainable, 'total': total}


def demonstrate_model():
    """Demonstrate the model architecture and usage."""
    
    print_separator("BIRDNET GEOMODEL ARCHITECTURE DEMONSTRATION")
    
    # Example dataset dimensions
    n_species = 1000  # Number of unique bird species
    n_env_features = 50  # Number of environmental features
    batch_size = 32
    
    print("Dataset Configuration:")
    print(f"  Number of species: {n_species:,}")
    print(f"  Environmental features: {n_env_features}")
    print(f"  Batch size: {batch_size}")
    
    # Create models of different sizes
    print_separator("MODEL CONFIGURATIONS")
    
    sizes = ['small', 'medium', 'large']
    models = {}
    
    for size in sizes:
        model = create_model(
            n_species=n_species,
            n_env_features=n_env_features,
            model_size=size
        )
        models[size] = model
        params = count_parameters(model)
        
        print(f"{size.upper()} Model:")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Size estimate: ~{params['total'] * 4 / 1024 / 1024:.1f} MB (float32)")
        print()
    
    # Use medium model for detailed demonstration
    model = models['medium']
    
    print_separator("MODEL ARCHITECTURE DETAILS")
    print("Full architecture:\n")
    print(model)
    
    # Create sample input data
    print_separator("FORWARD PASS DEMONSTRATION")
    
    print("Creating sample input data...")
    # Simulate sinusoidal encoded coordinates (batch_size, 4)
    coordinates = torch.randn(batch_size, 4)
    
    # Simulate sinusoidal encoded weeks (batch_size, 2)
    week = torch.randn(batch_size, 2)
    
    print("\nInput shapes:")
    print(f"  Coordinates: {coordinates.shape} (sinusoidal encoded lat/lon)")
    print(f"  Week: {week.shape} (sinusoidal encoded week)")
    
    # Forward pass (training mode with environmental prediction)
    print("\nForward pass (training mode)...")
    model.train()
    output_train = model(coordinates, week, return_env=True)
    
    print("\nOutput shapes (training mode):")
    print(f"  Species logits: {output_train['species_logits'].shape}")
    print(f"  Environmental predictions: {output_train['env_pred'].shape}")
    
    # Forward pass (inference mode without environmental prediction)
    print("\nForward pass (inference mode)...")
    model.eval()
    output_inference = model(coordinates, week, return_env=False)
    
    print("\nOutput shapes (inference mode):")
    print(f"  Species logits: {output_inference['species_logits'].shape}")
    print(f"  Environmental predictions: {'Not computed' if 'env_pred' not in output_inference else output_inference['env_pred'].shape}")
    
    # Demonstrate species prediction
    print_separator("SPECIES PREDICTION")
    
    probabilities = model.get_species_probabilities(coordinates, week)
    predictions = model.predict_species(coordinates, week, threshold=0.5)
    
    print(f"Species probabilities shape: {probabilities.shape}")
    print(f"Species predictions shape: {predictions.shape}")
    
    print("\nExample predictions for first sample:")
    sample_idx = 0
    predicted_species_indices = torch.where(predictions[sample_idx] == 1)[0]
    top_5_probs, top_5_indices = torch.topk(probabilities[sample_idx], k=5)
    
    print(f"  Total species predicted present: {len(predicted_species_indices)}")
    print("  Top 5 species by probability:")
    for i, (idx, prob) in enumerate(zip(top_5_indices, top_5_probs), 1):
        print(f"    {i}. Species {idx.item()}: {prob.item():.4f}")
    
    # Demonstrate loss computation
    print_separator("LOSS COMPUTATION")
    
    # Create dummy targets
    species_targets = torch.randint(0, 2, (batch_size, n_species)).float()
    env_targets = torch.randn(batch_size, n_env_features)
    
    print("Creating sample targets...")
    print(f"  Species targets shape: {species_targets.shape}")
    print(f"  Environmental targets shape: {env_targets.shape}")
    
    # Compute positive weights for handling class imbalance
    pos_weights = compute_pos_weights(species_targets)
    print(f"\nPositive class weights computed: {pos_weights.shape}")
    print(f"  Weight range: {pos_weights.min():.2f} - {pos_weights.max():.2f}")
    print(f"  Mean weight: {pos_weights.mean():.2f}")
    
    # Initialize loss function
    criterion = MultiTaskLoss(
        species_weight=1.0,
        env_weight=0.5,
        pos_weight=pos_weights
    )
    
    print("\nLoss configuration:")
    print("  Species weight: 1.0 (primary task)")
    print("  Environmental weight: 0.5 (auxiliary task)")
    print("  Using positive class weights: Yes")
    
    # Compute loss
    targets = {
        'species': species_targets,
        'env_features': env_targets
    }
    
    losses = criterion(output_train, targets)
    
    print("\nLoss values:")
    print(f"  Species loss: {losses['species'].item():.4f}")
    print(f"  Environmental loss: {losses['env'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    
    # Show what happens during inference (no env loss)
    losses_inference = criterion(output_inference, targets, compute_env_loss=False)
    print("\nInference mode (no environmental loss):")
    print(f"  Species loss: {losses_inference['species'].item():.4f}")
    print(f"  Total loss: {losses_inference['total'].item():.4f}")
    
    print_separator("MODEL COMPONENTS")
    
    print("Encoder:")
    print("  Input: Concatenated coordinates (4) + week (2) = 6 features")
    print(f"  Architecture: {model.encoder.encoder}")
    print(f"  Output dimension: {model.encoder.output_dim}")
    
    print("\nSpecies Prediction Head:")
    print(f"  Input dimension: {model.encoder.output_dim}")
    print(f"  Output dimension: {n_species} (number of species)")
    print("  Task: Multi-label binary classification")
    print("  Activation: Sigmoid (applied after logits)")
    
    print("\nEnvironmental Prediction Head:")
    print(f"  Input dimension: {model.encoder.output_dim}")
    print(f"  Output dimension: {n_env_features}")
    print("  Task: Regression (auxiliary)")
    print("  Purpose: Regularization + learning spatial patterns")
    
    print_separator("SUMMARY")
    
    print("Model Overview:")
    print("  ✓ Multi-task architecture with shared encoder")
    print("  ✓ Species prediction (primary): Multi-label classification")
    print("  ✓ Environmental prediction (auxiliary): Regression")
    print("  ✓ Sinusoidal encoding for coordinates and weeks")
    print("  ✓ Batch normalization and dropout for regularization")
    print("  ✓ Flexible model sizes (small/medium/large)")
    print("  ✓ Support for class imbalance via positive weights")
    
    print("\nTraining Pipeline:")
    print("  1. Encode coordinates and weeks sinusoidally")
    print("  2. Pass through shared encoder")
    print("  3. Predict species (primary) + environmental features (auxiliary)")
    print("  4. Compute weighted multi-task loss")
    print("  5. Backpropagate and update weights")
    
    print("\nInference Pipeline:")
    print("  1. Encode coordinates and weeks sinusoidally")
    print("  2. Pass through encoder → species head only")
    print("  3. Apply sigmoid to get probabilities")
    print("  4. Threshold to get binary predictions")
    
    print_separator()
    
    return model, criterion


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  BirdNET Geomodel - Model Architecture Demonstration")
    print("="*80)
    
    try:
        model, criterion = demonstrate_model()
        print("\n✓ Model demonstration completed successfully!")
        print(f"\nModel ready for training with {count_parameters(model)['trainable']:,} trainable parameters")
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
