"""Test PROCESSED (normalized) patterns"""
import requests
import json

def test_pattern(pattern_file, name, expected):
    with open(pattern_file, 'r') as f:
        pattern = json.load(f)
    
    print("=" * 70)
    print(f"Testing: {name}")
    print("=" * 70)
    
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": pattern}
    )
    
    if response.status_code == 200:
        result = response.json()
        predicted = result['prediction']
        confidence = result['confidence'] * 100
        
        status = "✅ CORRECT" if predicted == expected else "❌ WRONG"
        
        print(f"\nExpected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Status: {status}")
        
        print(f"\nGating Weights:")
        for expert, weight in result['gating_weights'].items():
            print(f"  {expert}: {weight:.4f}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

print("\n" + "=" * 70)
print("🧪 Testing NORMALIZED Patterns (Post-Preprocessing)")
print("=" * 70)

test_pattern('realistic_attack_processed.json', 'Attack (Normalized)', 'Attack')
print()
test_pattern('realistic_normal_processed.json', 'Normal (Normalized)', 'Normal')

print("\n" + "=" * 70)
print("✅ Test Complete!")
print("=" * 70)
