"""Extract real examples for Streamlit dashboard"""
import numpy as np
import json

# Load data
X = np.load('data/processed/cicids/X.npy')
y = np.load('data/processed/cicids/y.npy')

with open('data/processed/cicids/metadata.json', 'r') as f:
    metadata = json.load(f)

features = metadata['numeric_columns']

# Get diverse attack samples (DDoS is label=1)
attack_indices = np.where(y == 1)[0]
ddos_example = {f: float(X[attack_indices[10]][i]) for i, f in enumerate(features)}

# Get normal sample
normal_indices = np.where(y == 0)[0]
normal_example = {f: float(X[normal_indices[10]][i]) for i, f in enumerate(features)}

# Save
with open('ddos_example.json', 'w') as f:
    json.dump(ddos_example, f, indent=2)

with open('normal_example.json', 'w') as f:
    json.dump(normal_example, f, indent=2)

print("✅ Saved ddos_example.json and normal_example.json")
print(f"   (Real samples from processed CICIDS data)")
