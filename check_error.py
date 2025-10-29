"""Quick error check"""
import requests
import json

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "features": {
                "Flow Duration": 1200,
                "Total Fwd Packets": 250,
                "SYN Flag Count": 250
            }
        }
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
