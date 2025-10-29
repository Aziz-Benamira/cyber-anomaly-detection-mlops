"""Fix Port Scan example by extracting from actual PortScan labeled data"""
import pandas as pd
import numpy as np
import json

print("=" * 70)
print("🔧 Fixing Port Scan Example")
print("=" * 70)

# Load scaler stats and metadata
with open('data/processed/cicids/scaler_stats.json', 'r') as f:
    scaler_stats = json.load(f)

with open('data/processed/cicids/metadata.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['numeric_columns']
means = np.array(scaler_stats['mean'])
stds = np.array(scaler_stats['std'])

# Load Port Scan data
print("\n📁 Loading Port Scan file...")
df = pd.read_csv('data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
print(f"Total rows: {len(df):,}")
print(f"Labels: {df[' Label'].unique()}")

portscan_rows = df[df[' Label'] == 'PortScan']
print(f"Port Scan samples: {len(portscan_rows):,}")

# Get sample 100
sample = portscan_rows.iloc[100]

# Normalize
normalized = {}
for i, feat in enumerate(feature_names):
    # Try with and without leading space
    col = ' ' + feat if ' ' + feat in sample.index else feat
    if col in sample.index:
        val = float(sample[col]) if pd.notna(sample[col]) else 0.0
        normalized[feat] = (val - means[i]) / (stds[i] + 1e-8)
    else:
        normalized[feat] = 0.0

# Save
with open('port_scan.json', 'w') as f:
    json.dump(normalized, f, indent=2)

print(f"\n✅ Saved port_scan.json ({len(normalized)} features)")
print(f"Source label: {sample[' Label']}")

# Test prediction
import requests
result = requests.post('http://localhost:8000/predict', json={'features': normalized})
pred = result.json()
print(f"\n🔮 Prediction: {pred['prediction']}")
print(f"Confidence: {pred['confidence']*100:.2f}%")
print("=" * 70)
