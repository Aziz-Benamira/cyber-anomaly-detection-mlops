"""
Restart API and Test
====================
"""
import subprocess
import time
import requests
import json
import sys

print("=" * 70)
print("🔄 Restarting API Server")
print("=" * 70)

# Kill existing server on port 8000
print("\n1️⃣ Stopping existing server...")
try:
    subprocess.run(
        ["powershell", "-Command", "Get-Process | Where-Object {$_.ProcessName -eq 'python'} | Where-Object {$_.CommandLine -like '*uvicorn*'} | Stop-Process -Force"],
        capture_output=True,
        timeout=5
    )
    print("✅ Stopped")
except:
    print("⚠️  No running server found or error stopping")

time.sleep(2)

# Start new server
print("\n2️⃣ Starting fresh server...")
print("Command: uvicorn src.serving.api:app --reload --port 8000\n")

server = subprocess.Popen(
    ["uvicorn", "src.serving.api:app", "--reload", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=r"C:\Users\benam\Downloads\cyber-anomaly-detection-mlops"
)

print("⏳ Waiting 10 seconds for startup...")
time.sleep(10)

# Test
print("\n3️⃣ Testing API...")
print("-" * 70)

BASE_URL = "http://localhost:8000"

try:
    # Health check
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"Health Check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Single prediction
    print("\n" + "-" * 70)
    print("Testing prediction...")
    
    payload = {
        "features": {
            "Flow Duration": 1200,
            "Total Fwd Packets": 250,
            "SYN Flag Count": 250,
            "ACK Flag Count": 0
        },
        "handle_missing": True
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ SUCCESS!")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Inference Time: {result['inference_time_ms']:.2f}ms")
        
        if result.get('gating_weights'):
            print("\nGating Weights:")
            for expert, weight in result['gating_weights'].items():
                print(f"  {expert}: {weight:.4f}")
    else:
        print(f"\n❌ Error: {response.json()}")
        
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    print("\nCheck if server is running:")
    print("  uvicorn src.serving.api:app --reload --port 8000")

print("\n" + "=" * 70)
print("📚 API Docs: http://localhost:8000/docs")
print("=" * 70)
print("\nPress Ctrl+C to stop the server")

try:
    server.wait()
except KeyboardInterrupt:
    print("\n\n🛑 Stopping...")
    server.terminate()
    print("✅ Done")
