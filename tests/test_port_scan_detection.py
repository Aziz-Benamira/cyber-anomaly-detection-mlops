"""Test multiple port scan samples to see model behavior"""
import pandas as pd
import numpy as np
import json
import requests

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

print("=" * 70)
print(f'🔍 Testing Port Scan Detection (Total samples: {len(ps):,})')
print("=" * 70)

attack_count = 0
for idx in [10, 50, 100, 500, 1000, 5000, 10000, 50000]:
    sample = ps.iloc[idx]
    norm = {}
    for i, feat in enumerate(features):
        col = ' ' + feat if ' ' + feat in sample.index else feat
        val = float(sample[col]) if pd.notna(sample[col]) and col in sample.index else 0.0
        norm[feat] = (val - means[i]) / (stds[i] + 1e-8)
    
    r = requests.post('http://localhost:8000/predict', json={'features': norm})
    pred = r.json()
    
    status = "🔴" if pred['prediction'] == 'Attack' else "🟢"
    print(f"{status} Sample #{idx:6d}: {pred['prediction']:8s} ({pred['confidence']*100:5.1f}%)")
    
    if pred['prediction'] == 'Attack':
        attack_count += 1

print("=" * 70)
print(f"Detected as Attack: {attack_count}/8 samples")
print("=" * 70)
