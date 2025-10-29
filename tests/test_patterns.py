"""
Quick Test - Verify Example Patterns
=====================================
"""
import requests
import json

API_URL = "http://localhost:8000"

# Test patterns from streamlit_app.py
patterns = {
    "DDoS Attack": {
        "Flow Duration": 12000000,
        "Total Fwd Packets": 180000,
        "Total Backward Packets": 0,
        "SYN Flag Count": 180000,
        "ACK Flag Count": 0,
        "Average Packet Size": 40.0
    },
    "Normal Traffic": {
        "Flow Duration": 120000000,
        "Total Fwd Packets": 950,
        "Total Backward Packets": 850,
        "SYN Flag Count": 1,
        "ACK Flag Count": 1795,
        "Average Packet Size": 833.33
    }
}

print("=" * 70)
print("🧪 Testing Example Patterns")
print("=" * 70)

for name, features in patterns.items():
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    payload = {"features": features, "handle_missing": True}
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            prediction = result['prediction']
            confidence = result['confidence'] * 100
            
            # Check if correct
            if name == "DDoS Attack":
                expected = "Attack"
            else:
                expected = "Normal"
            
            is_correct = prediction == expected
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            
            print(f"\nPattern: {name}")
            print(f"Expected: {expected}")
            print(f"Predicted: {prediction}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Status: {status}")
            
            if result.get('gating_weights'):
                print(f"\nExpert Weights:")
                for expert, weight in result['gating_weights'].items():
                    print(f"  {expert}: {weight:.4f}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
    
    except Exception as e:
        print(f"❌ Exception: {e}")

print("\n" + "=" * 70)
print("✅ Test Complete!")
print("=" * 70)
