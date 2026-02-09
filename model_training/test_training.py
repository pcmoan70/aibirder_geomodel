"""
Quick test of the training pipeline with a small number of epochs.

This script tests that all components work together correctly.
"""

import subprocess
import sys

if __name__ == "__main__":
    # Run training for just 2 epochs to test the pipeline
    cmd = [
        sys.executable,
        "c:/Entwicklung/BirdNET/geomodel/model_training/train.py",
        "--num_epochs", "2",
        "--batch_size", "64",
        "--model_size", "small",
        "--save_every", "1"
    ]
    
    print("Testing training pipeline with 2 epochs...")
    print(" ".join(cmd))
    print()
    
    subprocess.run(cmd)
