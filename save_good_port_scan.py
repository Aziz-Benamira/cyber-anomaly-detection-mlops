"""Save the Port Scan sample that actually gets detected as Attack"""
import pandas as pd
import numpy as np
import json

# Load metadata and scaler
with open('data/processed/cicids/metadata.json') as f:
    meta = json.load(f)
with open('data/processed/cicids/scaler_stats.json') as f:
    scaler = json.load(f)

features = meta['numeric_columns']
means = np.array(scaler['mean'])
stds = np.array(scaler['std'])

# Load port scan data
df = pd.read_csv('data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
ps = df[df[' Label'] == 'PortScan']

# Use sample #10000 which gets detected as Attack
sample = ps.iloc[10000]
norm = {}
for i, feat in enumerate(features):
    col = ' ' + feat if ' ' + feat in sample.index else feat
    val = float(sample[col]) if pd.notna(sample[col]) and col in sample.index else 0.0
    norm[feat] = (val - means[i]) / (stds[i] + 1e-8)

# Save
with open('port_scan.json', 'w') as f:
    json.dump(norm, f, indent=2)

print("✅ Saved port_scan.json (sample #10000 - detected as Attack)")

# Verify
import requests
r = requests.post('http://localhost:8000/predict', json={'features': norm})
pred = r.json()
print(f"🔮 Prediction: {pred['prediction']} ({pred['confidence']*100:.1f}%)")
