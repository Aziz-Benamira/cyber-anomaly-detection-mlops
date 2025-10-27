"""
Helper script to create mini balanced datasets for faster experimentation.

This is useful for:
- Portfolio/demo projects (fast training)
- Local development with limited GPU
- Quick iterations during model development

Usage:
    python scripts/create_mini_datasets.py --dataset CICIDS --samples 50000
    python scripts/create_mini_datasets.py --dataset UNSW --samples 30000
    python scripts/create_mini_datasets.py --all --samples 40000
"""

import subprocess
import sys
from pathlib import Path


def create_mini_dataset(dataset_name: str, max_samples: int = 50000):
    """Create a mini balanced dataset"""
    print(f"\n{'='*60}")
    print(f"Creating MINI {dataset_name} dataset ({max_samples} samples)")
    print(f"{'='*60}\n")
    
    # Update params.yaml temporarily
    import yaml
    params_path = "params.yaml"
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    params["data"]["dataset"] = dataset_name
    
    with open(params_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False)
    
    # Run preprocessing with mini flag
    cmd = [
        sys.executable, "-m", "src.data.preprocess",
        "--params", params_path,
        "--out", "data/processed",
        "--mini",
        "--max-samples", str(max_samples)
    ]
    
    result = subprocess.run(cmd, check=True)
    
    print(f"\n✅ Mini {dataset_name} dataset created successfully!")
    print(f"   Location: data/processed/{dataset_name.lower()}/")
    
    return result.returncode == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create mini datasets for faster training")
    parser.add_argument("--dataset", choices=["CICIDS", "UNSW", "cicids", "unsw"], 
                        help="Dataset to process (CICIDS or UNSW)")
    parser.add_argument("--all", action="store_true", help="Process both datasets")
    parser.add_argument("--samples", type=int, default=50000,
                        help="Max samples per dataset (default: 50000)")
    
    args = parser.parse_args()
    
    if args.all:
        create_mini_dataset("CICIDS", args.samples)
        create_mini_dataset("UNSW", args.samples)
    elif args.dataset:
        create_mini_dataset(args.dataset.upper(), args.samples)
    else:
        parser.print_help()
        print("\n⚠️  Please specify --dataset or --all")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 All done! You can now train with:")
    print("   python -m src.models.train_tabular --stage pretrain")
    print("   python -m src.models.train_tabular --stage finetune")
    print("="*60)
