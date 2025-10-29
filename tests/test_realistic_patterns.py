"""Test realistic patterns extracted from actual CICIDS data"""
import requests
import json

def test_pattern(pattern_file, pattern_name, expected_label):
    with open(pattern_file, 'r') as f:
        pattern = json.load(f)
    
    print("=" * 70)
    print(f"Testing: {pattern_name}")
    print("=" * 70)
    
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": pattern}
    )
    
    if response.status_code == 200:
        result = response.json()
        predicted = result['prediction']
        confidence = result['confidence'] * 100
        
        status = "✅ CORRECT" if predicted == expected_label else "❌ WRONG"
        
        print(f"\nPattern: {pattern_name}")
        print(f"Expected: {expected_label}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Status: {status}")
        
        print(f"\nExpert Weights:")
        for expert, weight in result['gating_weights'].items():
            print(f"  {expert}: {weight:.4f}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

print("=" * 70)
print("🧪 Testing REALISTIC Patterns from Actual CICIDS Data")
print("=" * 70)

test_pattern('realistic_ddos_pattern.json', 'DDoS Attack (Real Data)', 'Attack')
print()
test_pattern('realistic_normal_pattern.json', 'Normal Traffic (Real Data)', 'Normal')

print("\n" + "=" * 70)
print("✅ Test Complete!")
print("=" * 70)
