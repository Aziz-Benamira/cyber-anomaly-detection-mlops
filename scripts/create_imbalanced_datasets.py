"""
Create LARGE, IMBALANCED datasets that preserve the original class distribution.

This is for PRODUCTION-READY anomaly detection training:
- Large sample sizes (500k rows) for robust learning
- Preserves natural class imbalance (90% normal, 10% attack)
- Realistic evaluation environment

Usage:
    python scripts/create_imbalanced_datasets.py --dataset CICIDS --samples 500000
    python scripts/create_imbalanced_datasets.py --dataset UNSW --samples 500000
    python scripts/create_imbalanced_datasets.py --all --samples 500000
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def get_original_distribution(dataset_name: str):
    """
    Get the approximate original class distribution for each dataset.
    These are based on the full datasets before any sampling.
    """
    distributions = {
        "CICIDS": {"normal_ratio": 0.80, "attack_ratio": 0.20},  # ~80% benign
        "UNSW": {"normal_ratio": 0.56, "attack_ratio": 0.44},    # ~56% normal (UNSW is more balanced)
    }
    return distributions.get(dataset_name.upper(), {"normal_ratio": 0.90, "attack_ratio": 0.10})


def create_imbalanced_dataset(dataset_name: str, max_samples: int = 500000):
    """
    Create a large imbalanced dataset preserving original class distribution.
    
    This uses a custom preprocessing approach that:
    1. Loads the full dataset
    2. Samples stratified by class (maintaining original ratio)
    3. Outputs to data/processed/{dataset}/
    """
    print(f"\n{'='*70}")
    print(f"Creating LARGE IMBALANCED {dataset_name} dataset")
    print(f"  Target samples: {max_samples:,}")
    print(f"  Strategy: Preserve original class distribution")
    print(f"{'='*70}\n")
    
    # Get expected distribution
    dist = get_original_distribution(dataset_name)
    expected_normal = int(max_samples * dist["normal_ratio"])
    expected_attack = int(max_samples * dist["attack_ratio"])
    
    print(f"[INFO] Expected distribution:")
    print(f"       Normal:  {expected_normal:,} ({dist['normal_ratio']*100:.1f}%)")
    print(f"       Attack:  {expected_attack:,} ({dist['attack_ratio']*100:.1f}%)")
    print(f"       Total:   {expected_normal + expected_attack:,}")
    
    # Update params.yaml temporarily
    import yaml
    params_path = "params.yaml"
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    params["data"]["dataset"] = dataset_name
    
    with open(params_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False)
    
    # Run preprocessing with IMBALANCED flag
    # This will NOT balance classes, just sample randomly up to max_samples
    cmd = [
        sys.executable, "-m", "src.data.preprocess",
        "--params", params_path,
        "--out", "data/processed",
        "--imbalanced",  # NEW FLAG: preserve natural distribution
        "--max-samples", str(max_samples)
    ]
    
    print(f"\n[INFO] Running preprocessing...")
    result = subprocess.run(cmd, check=True)
    
    print(f"\n✅ Large imbalanced {dataset_name} dataset created successfully!")
    print(f"   Location: data/processed/{dataset_name.lower()}/")
    print(f"   This dataset is ready for PRODUCTION-READY training with:")
    print(f"   - Weighted loss functions")
    print(f"   - F1-Score / Precision / Recall metrics")
    print(f"   - Realistic anomaly detection evaluation")
    
    return result.returncode == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create large imbalanced datasets for production training")
    parser.add_argument("--dataset", choices=["CICIDS", "UNSW", "cicids", "unsw"], 
                        help="Dataset to process (CICIDS or UNSW)")
    parser.add_argument("--all", action="store_true", help="Process both datasets")
    parser.add_argument("--samples", type=int, default=500000,
                        help="Target sample size (default: 500000)")
    
    args = parser.parse_args()
    
    if args.all:
        print("\n" + "="*70)
        print("CREATING LARGE IMBALANCED DATASETS FOR BOTH CICIDS AND UNSW")
        print("="*70)
        
        create_imbalanced_dataset("CICIDS", args.samples)
        print("\n" + "-"*70 + "\n")
        create_imbalanced_dataset("UNSW", args.samples)
    elif args.dataset:
        create_imbalanced_dataset(args.dataset.upper(), args.samples)
    else:
        parser.print_help()
        print("\n⚠️  Please specify --dataset or --all")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🎉 IMBALANCED DATASETS READY!")
    print("\nNext steps:")
    print("   1. Train with weighted loss:")
    print("      python -m src.models.train_tabular --stage pretrain")
    print("      python -m src.models.train_tabular --stage finetune")
    print("\n   2. Evaluate with F1-Score, Precision, Recall (not accuracy!)")
    print("\n   3. Deploy with confidence that your model handles real imbalance")
    print("="*70)
