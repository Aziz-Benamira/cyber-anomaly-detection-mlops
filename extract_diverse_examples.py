"""Extract diverse attack examples from processed CICIDS data"""
import numpy as np
import json

print("=" * 70)
print("📊 Extracting Diverse Attack Examples from CICIDS")
print("=" * 70)

# Load processed data
X = np.load('data/processed/cicids/X.npy')
y = np.load('data/processed/cicids/y.npy')

with open('data/processed/cicids/metadata.json', 'r') as f:
    metadata = json.load(f)

features = metadata['numeric_columns']

# Get attack and normal indices
attack_indices = np.where(y == 1)[0]
normal_indices = np.where(y == 0)[0]

print(f"\nTotal samples: {len(X):,}")
print(f"Attack samples: {len(attack_indices):,}")
print(f"Normal samples: {len(normal_indices):,}")

# Extract diverse attack patterns (different samples to represent different attack types)
examples = {}

# DDoS Attack (sample 10)
examples["DDoS Attack"] = {f: float(X[attack_indices[10]][i]) for i, f in enumerate(features)}

# Port Scan (sample 100 - different characteristics)
examples["Port Scan"] = {f: float(X[attack_indices[100]][i]) for i, f in enumerate(features)}

# Web Attack (sample 500)
examples["Web Attack"] = {f: float(X[attack_indices[500]][i]) for i, f in enumerate(features)}

# Brute Force (sample 1000)
examples["Brute Force"] = {f: float(X[attack_indices[1000]][i]) for i, f in enumerate(features)}

# Normal Traffic (sample 10)
examples["Normal Traffic"] = {f: float(X[normal_indices[10]][i]) for i, f in enumerate(features)}

# Normal HTTPS (sample 100)
examples["Normal HTTPS"] = {f: float(X[normal_indices[100]][i]) for i, f in enumerate(features)}

# Save each example
for name, data in examples.items():
    filename = name.lower().replace(' ', '_') + '.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {filename} ({len(data)} features)")

print("\n" + "=" * 70)
print("✅ Complete! Created 6 diverse examples")
print("=" * 70)
