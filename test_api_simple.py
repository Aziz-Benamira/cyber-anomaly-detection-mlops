"""
Simple API Test
===============
Run this after starting the server with:
  uvicorn src.serving.api:app --port 8000
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("🧪 FastAPI Test Suite")
print("=" * 70)

# Wait a bit for server to be ready
print("\n⏳ Waiting for server...")
time.sleep(3)

# Test 1: Health Check
print("\n" + "=" * 70)
print("1️⃣ HEALTH CHECK")
print("=" * 70)
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"✅ Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Failed: {e}")
    print("\n⚠️  Make sure the API is running:")
    print("   uvicorn src.serving.api:app --port 8000")
    exit(1)

# Test 2: Model Info
print("\n" + "=" * 70)
print("2️⃣ MODEL INFO")
print("=" * 70)
try:
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"✅ Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 3: Single Prediction
print("\n" + "=" * 70)
print("3️⃣ SINGLE PREDICTION (Attack Pattern)")
print("=" * 70)

# Attack-like features: High SYN, no ACK (SYN flood pattern)
attack_features = {
    "Flow Duration": 1200,
    "Total Fwd Packets": 250,
    "Total Backward Packets": 0,
    "Flow IAT Mean": 4.8,
    "SYN Flag Count": 250,
    "ACK Flag Count": 0
}

payload = {
    "features": attack_features,
    "handle_missing": True
}

print("Request:")
print(json.dumps(payload, indent=2))

try:
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\n✅ Status: {response.status_code}")
    result = response.json()
    
    print("\nPrediction Results:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Inference Time: {result['inference_time_ms']:.2f}ms")
    
    if result.get('probabilities'):
        print(f"\nProbabilities:")
        for label, prob in result['probabilities'].items():
            print(f"    {label}: {prob:.4f}")
    
    if result.get('gating_weights'):
        print(f"\nGating Weights:")
        for expert, weight in result['gating_weights'].items():
            print(f"    {expert}: {weight:.4f}")
            
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: Normal Traffic Prediction
print("\n" + "=" * 70)
print("4️⃣ SINGLE PREDICTION (Normal Pattern)")
print("=" * 70)

# Normal-like features: Balanced flags, bidirectional
normal_features = {
    "Flow Duration": 5000,
    "Total Fwd Packets": 10,
    "Total Backward Packets": 8,
    "Flow IAT Mean": 500,
    "SYN Flag Count": 1,
    "ACK Flag Count": 15
}

payload = {"features": normal_features, "handle_missing": True}

try:
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    result = response.json()
    
    print(f"✅ Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")
            
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 5: Batch Prediction
print("\n" + "=" * 70)
print("5️⃣ BATCH PREDICTION (3 samples)")
print("=" * 70)

batch_payload = {
    "features": [attack_features, normal_features, attack_features],
    "handle_missing": True
}

try:
    response = requests.post(f"{BASE_URL}/predict_batch", json=batch_payload)
    result = response.json()
    
    print(f"✅ Status: {response.status_code}")
    print(f"\nBatch Results:")
    print(f"  Total samples: {len(result['predictions'])}")
    print(f"  Total time: {result['total_time_ms']:.2f}ms")
    print(f"  Average time: {result['average_time_ms']:.2f}ms")
    
    print(f"\nPredictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"    Sample {i}: {pred['prediction']} ({pred['confidence']:.2%})")
            
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 6: Metrics
print("\n" + "=" * 70)
print("6️⃣ METRICS")
print("=" * 70)
try:
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"✅ Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Failed: {e}")

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETED!")
print("=" * 70)
print(f"\n📚 Interactive API Docs: {BASE_URL}/docs")
print(f"🔍 ReDoc Documentation: {BASE_URL}/redoc")
print(f"📊 Health Check: {BASE_URL}/health")
print("\n" + "=" * 70)
