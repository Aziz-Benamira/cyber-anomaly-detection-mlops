"""Extract realistic patterns from PROCESSED CICIDS data (post-preprocessing)"""
import numpy as np
import json

print("=" * 70)
print("📊 Extracting Patterns from PROCESSED CICIDS Data")
print("=" * 70)

# Load processed data
X = np.load('data/processed/cicids/X.npy')
y = np.load('data/processed/cicids/y.npy')

# Load metadata for feature names
with open('data/processed/cicids/metadata.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['numeric_columns']

print(f"\nLoaded {len(X)} samples")
print(f"Features: {len(feature_names)}")
print(f"Attack samples: {(y == 1).sum():,}")
print(f"Normal samples: {(y == 0).sum():,}")

# Extract attack pattern (median of first 100 attacks)
attack_indices = np.where(y == 1)[0][:100]
attack_pattern = np.median(X[attack_indices], axis=0)

# Extract normal pattern (median of first 100 normal)
normal_indices = np.where(y == 0)[0][:100]
normal_pattern = np.median(X[normal_indices], axis=0)

# Create dictionaries
attack_dict = {name: float(val) for name, val in zip(feature_names, attack_pattern)}
normal_dict = {name: float(val) for name, val in zip(feature_names, normal_pattern)}

# Save
with open('realistic_attack_processed.json', 'w') as f:
    json.dump(attack_dict, f, indent=2)

with open('realistic_normal_processed.json', 'w') as f:
    json.dump(normal_dict, f, indent=2)

print("\n✅ Saved patterns:")
print("   - realistic_attack_processed.json (72 features, NORMALIZED)")
print("   - realistic_normal_processed.json (72 features, NORMALIZED)")

print("\n📊 Sample values (Attack pattern):")
for feat in ['Flow Duration', 'Total Fwd Packets', 'SYN Flag Count', 'Flow Bytes/s']:
    print(f"   {feat}: {attack_dict.get(feat, 'N/A'):.4f}")

print("\n📊 Sample values (Normal pattern):")
for feat in ['Flow Duration', 'Total Fwd Packets', 'SYN Flag Count', 'Flow Bytes/s']:
    print(f"   {feat}: {normal_dict.get(feat, 'N/A'):.4f}")
