"""Generate scaler statistics for inference normalization"""
import numpy as np
import json
from pathlib import Path
import joblib

def extract_scaler_stats(dataset='cicids'):
    """Extract mean and std from preprocessor for inference"""
    
    dataset_dir = f"data/processed/{dataset.lower()}"
    preprocessor_path = f"{dataset_dir}/preprocessor.joblib"
    
    print(f"Loading preprocessor from: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    
    # Get the scaler from the ColumnTransformer
    # The numeric pipeline is the second transformer (index 1)
    numeric_transformer = preprocessor.transformers_[1][1]
    scaler = numeric_transformer.named_steps['scaler']
    
    # Extract mean and std
    mean = scaler.mean_.tolist()
    std = scaler.scale_.tolist()  # scale_ is the std
    
    stats = {
        'mean': mean,
        'std': std,
        'n_features': len(mean)
    }
    
    # Save to JSON
    output_path = f"{dataset_dir}/scaler_stats.json"
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Saved scaler stats to: {output_path}")
    print(f"   Features: {len(mean)}")
    print(f"   Mean range: [{min(mean):.4f}, {max(mean):.4f}]")
    print(f"   Std range: [{min(std):.4f}, {max(std):.4f}]")
    
    return stats

if __name__ == "__main__":
    print("=" * 70)
    print("📊 Extracting Scaler Statistics for Inference")
    print("=" * 70)
    
    # Extract for CICIDS
    stats = extract_scaler_stats('cicids')
    
    print("\n" + "=" * 70)
    print("✅ Complete! API can now normalize input features correctly")
    print("=" * 70)
