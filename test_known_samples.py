"""Test with a single known attack sample from processed data"""
import numpy as np
import json
import requests

# Load data
X = np.load('data/processed/cicids/X.npy')
y = np.load('data/processed/cicids/y.npy')

with open('data/processed/cicids/metadata.json', 'r') as f:
    metadata = json.load(f)

features = metadata['numeric_columns']

# Get first attack sample (y=1)
attack_idx = np.where(y == 1)[0][0]
attack_sample = {f: float(X[attack_idx][i]) for i, f in enumerate(features)}

# Get first normal sample (y=0)
normal_idx = np.where(y == 0)[0][0]
normal_sample = {f: float(X[normal_idx][i]) for i, f in enumerate(features)}

print("=" * 70)
print("🧪 Testing with ACTUAL Training Data Samples")
print("=" * 70)

print("\n[TEST 1] Known ATTACK sample (y=1):")
r = requests.post('http://localhost:8000/predict', json={'features': attack_sample})
result = r.json()
print(f"  TRUE LABEL: Attack")
print(f"  PREDICTED: {result['prediction']}")
print(f"  CONFIDENCE: {result['confidence'] * 100:.2f}%")
print(f"  STATUS: {'✅ CORRECT' if result['prediction'] == 'Attack' else '❌ WRONG'}")

print("\n[TEST 2] Known NORMAL sample (y=0):")
r = requests.post('http://localhost:8000/predict', json={'features': normal_sample})
result = r.json()
print(f"  TRUE LABEL: Normal")
print(f"  PREDICTED: {result['prediction']}")
print(f"  CONFIDENCE: {result['confidence'] * 100:.2f}%")
print(f"  STATUS: {'✅ CORRECT' if result['prediction'] == 'Normal' else '❌ WRONG'}")

print("\n" + "=" * 70)
if result['prediction'] == 'Attack':
    print("⚠️  LABELS ARE FLIPPED! Model trained with inverted labels!")
else:
    print("✅ Labels are correct - investigating why attacks predict as Normal...")
print("=" * 70)
