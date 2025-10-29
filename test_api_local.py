"""
Quick Local API Test (No Docker Required)
==========================================
Tests the FastAPI app using uvicorn locally
"""

import subprocess
import time
import requests
import json

print("=" * 70)
print("🧪 Testing FastAPI Locally (No Docker)")
print("=" * 70)

# Start the API server
print("\n1️⃣ Starting FastAPI server...")
print("Command: uvicorn src.serving.api:app --port 8000")
print("-" * 70)

# Start server in background
server = subprocess.Popen(
    ["uvicorn", "src.serving.api:app", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("⏳ Waiting for server to start...")
time.sleep(5)  # Wait for startup

BASE_URL = "http://localhost:8000"

try:
    # Test 1: Health Check
    print("\n2️⃣ Testing Health Check")
    print("-" * 70)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Model Info
    print("\n3️⃣ Testing Model Info")
    print("-" * 70)
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 3: Single Prediction
    print("\n4️⃣ Testing Single Prediction")
    print("-" * 70)
    
    # Sample attack-like features (high SYN flags, no ACK)
    sample_features = {
        "Flow Duration": 1200,
        "Total Fwd Packets": 250,
        "Total Backward Packets": 0,
        "Flow IAT Mean": 4.8,
        "Flow IAT Max": 100,
        "Fwd IAT Mean": 4.8,
        "Bwd IAT Mean": 0,
        "SYN Flag Count": 250,
        "ACK Flag Count": 0,
        "PSH Flag Count": 0,
        "FIN Flag Count": 0
    }
    
    payload = {
        "features": sample_features,
        "handle_missing": True,
        "missing_strategy": "zero"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 4: Batch Prediction
    print("\n5️⃣ Testing Batch Prediction")
    print("-" * 70)
    
    batch_payload = {
        "features": [sample_features, sample_features],
        "handle_missing": True
    }
    
    response = requests.post(f"{BASE_URL}/predict_batch", json=batch_payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total samples: {len(result['predictions'])}")
    print(f"Average inference time: {result['average_time_ms']:.2f}ms")
    print(f"First prediction: {result['predictions'][0]['prediction']}")
    
    # Test 5: Metrics
    print("\n6️⃣ Testing Metrics Endpoint")
    print("-" * 70)
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print(f"\n📚 Interactive docs: {BASE_URL}/docs")
    print(f"🔍 Redoc: {BASE_URL}/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    # Keep server running
    server.wait()
    
except requests.exceptions.ConnectionError:
    print("\n❌ Error: Could not connect to API")
    print("The server might not have started properly.")
    print("\nTry running manually:")
    print("  uvicorn src.serving.api:app --reload --port 8000")
    
except KeyboardInterrupt:
    print("\n\n🛑 Stopping server...")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    
finally:
    server.terminate()
    server.wait()
    print("✅ Server stopped")
