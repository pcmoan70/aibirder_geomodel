# BirdNET Geomodel Training

Complete training pipeline for the spatiotemporal bird species occurrence prediction model.

## Training Script

Train the model using:

```bash
python model_training/train.py [OPTIONS]
```

### Basic Usage

```bash
# Train with default settings (medium model, 50 epochs)
python model_training/train.py

# Train small model for 20 epochs
python model_training/train.py --model_size small --num_epochs 20

# Train with custom batch size and learning rate
python model_training/train.py --batch_size 128 --lr 0.0005

# Resume from checkpoint
python model_training/train.py --resume ./checkpoints/20260209_154505/checkpoint_best.pt
```

### Command Line Arguments

#### Data Arguments
- `--data_path`: Path to parquet data file (default: `./outputs/global_350km_ee_gbif.parquet`)

#### Model Arguments
- `--model_size`: Model size configuration - `small`, `medium`, or `large` (default: `medium`)

#### Training Arguments
- `--batch_size`: Batch size for training (default: `256`)
- `--num_epochs`: Number of training epochs (default: `50`)
- `--lr`: Learning rate (default: `0.001`)
- `--weight_decay`: Weight decay / L2 regularization (default: `1e-5`)
- `--species_weight`: Weight for species loss (default: `1.0`)
- `--env_weight`: Weight for environmental loss (default: `0.5`)

#### Data Split Arguments
- `--test_size`: Test set proportion (default: `0.2`)
- `--val_size`: Validation set proportion of training data (default: `0.1`)

#### Checkpoint Arguments
- `--checkpoint_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--resume`: Path to checkpoint to resume from (default: `None`)
- `--save_every`: Save checkpoint every N epochs (default: `5`)

#### Device Arguments
- `--device`: Device to train on - `auto`, `cuda`, or `cpu` (default: `auto`)
- `--num_workers`: Number of data loading workers (default: `0`)

## Model Sizes

| Size   | Parameters | Memory  | Use Case |
|--------|-----------|---------|----------|
| Small  | ~1.4M     | ~5.5 MB  | Quick experiments, limited resources |
| Medium | ~1.1M     | ~4.3 MB  | Default, balanced performance |
| Large  | ~3.4M     | ~13 MB   | Best performance, more data |

## Training Output

Training creates a timestamped checkpoint directory with:
- `checkpoint_latest.pt` - Latest model state
- `checkpoint_best.pt` - Best model (lowest validation loss)
- `checkpoint_epoch_N.pt` - Checkpoints every N epochs
- `training_history.json` - Loss history for all epochs

## Example Training Run

```bash
python model_training/train.py \
  --model_size medium \
  --batch_size 256 \
  --num_epochs 50 \
  --lr 0.001 \
  --species_weight 1.0 \
  --env_weight 0.5 \
  --device auto
```

Expected output:
```
================================================================================
  BirdNET Geomodel Training
================================================================================

Configuration:
  Data: ./outputs/global_350km_ee_gbif.parquet
  Model size: medium
  Batch size: 256
  Epochs: 50
  Learning rate: 0.001
  Device: cuda

...

Epoch 1/50 Summary:
  Train - Loss: 0.3304, Species: 0.0059, Env: 0.6490
  Val   - Loss: 0.2143, Species: 0.0012, Env: 0.4262
✓ Saved best model (val_loss: 0.2143)

...
```

## Multi-Task Loss

The model optimizes a weighted combination of two losses:

**Total Loss = species_weight × Species Loss + env_weight × Environmental Loss**

- **Species Loss** (Primary): Binary cross-entropy for multi-label classification
- **Environmental Loss** (Auxiliary): MSE for environmental feature regression

The environmental task helps the model learn better spatial representations during training.

## Checkpoints

Checkpoint files contain:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'best_val_loss': float,
    'history': dict
}
```

Load a checkpoint:
```python
checkpoint = torch.load('checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Tips

1. **Start small**: Test with `--model_size small --num_epochs 5` first
2. **Monitor losses**: Species loss should be the primary metric
3. **Adjust weights**: If environmental loss dominates, reduce `--env_weight`
4. **Learning rate**: Reduce if loss diverges, increase if training is too slow
5. **GPU training**: Use `--device cuda` for 10-20x speedup on compatible hardware
