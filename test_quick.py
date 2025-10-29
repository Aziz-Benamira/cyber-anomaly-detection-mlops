import json
import requests
import time

print("Waiting for API to start...")
time.sleep(5)

# Test with DDoS example
with open('ddos_attack.json', 'r') as f:
    features = json.load(f)

print(f"\n✅ Loaded {len(features)} features from ddos_attack.json")
print(f"\n🔮 Making prediction...")

response = requests.post(
    'http://localhost:8000/predict', 
    json={'features': features},
    timeout=10
)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"\n🎯 Result:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Probabilities: {result['probabilities']}")
    print(f"  Gating Weights: {result['gating_weights']}")
    print(f"\n✅ SUCCESS! Predictions are working!")
else:
    print(f"❌ Error: {response.text}")
