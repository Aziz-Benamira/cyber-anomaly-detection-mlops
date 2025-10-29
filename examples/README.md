# 📋 Example Attack Patterns

This directory contains pre-processed example traffic patterns for testing the MoE anomaly detection model.

## Available Examples

### **Attack Patterns**

| File | Attack Type | Description |
|------|-------------|-------------|
| `ddos_attack.json` | DDoS | Distributed Denial of Service attack pattern |
| `port_scan.json` | Port Scan | Network port scanning activity |
| `web_attack.json` | Web Attack | Web application attack (SQL injection, XSS, etc.) |
| `brute_force.json` | Brute Force | Authentication brute force attempt |
| `ddos_example.json` | DDoS | Alternative DDoS pattern |
| `realistic_attack_processed.json` | Attack | Realistic attack from live traffic |
| `realistic_ddos_pattern.json` | DDoS | Realistic DDoS pattern |

### **Normal Traffic Patterns**

| File | Traffic Type | Description |
|------|--------------|-------------|
| `normal_traffic.json` | Normal HTTP | Regular HTTP web browsing |
| `normal_https.json` | Normal HTTPS | Secure web browsing |
| `normal_example.json` | Normal | General normal traffic |
| `realistic_normal_pattern.json` | Normal | Realistic normal traffic |
| `realistic_normal_processed.json` | Normal | Processed normal traffic |

## Usage

### **With Streamlit Dashboard**
```bash
streamlit run src/serving/dashboard.py --server.port 8501

# Then select example from dropdown and click "Analyze Traffic"
```

### **With API (cURL)**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/ddos_attack.json
```

### **With Python**
```python
import requests
import json

# Load example
with open('examples/ddos_attack.json', 'r') as f:
    features = json.load(f)

# Make prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={'features': features}
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## File Format

Each JSON file contains 72 normalized features:

```json
{
  "Flow Duration": 0.123456,
  "Total Fwd Packets": 1.234567,
  "Total Backward Packets": 0.987654,
  ...
  "Idle Mean": 0.456789
}
```

Features are already normalized using StandardScaler based on training data statistics.

## Expected Results

| Example | Expected Prediction | Expected Confidence |
|---------|-------------------|-------------------|
| `ddos_attack.json` | Attack | ~99.99% |
| `port_scan.json` | Attack | ~53-65% |
| `web_attack.json` | Attack | ~85-95% |
| `brute_force.json` | Attack | ~75-90% |
| `normal_traffic.json` | Normal | ~99%+ |
| `normal_https.json` | Normal | ~99%+ |

## Creating Custom Examples

To create your own example:

1. **Use Streamlit Dashboard:**
   - Select "Custom" mode
   - Edit the 72 features
   - Click "Analyze Traffic"
   - Export the pattern (if implemented)

2. **Manual Creation:**
   ```python
   import json
   
   # Create custom pattern
   custom_pattern = {
       "Flow Duration": 0.5,
       "Total Fwd Packets": 2.0,
       # ... all 72 features
   }
   
   # Save to file
   with open('examples/custom_attack.json', 'w') as f:
       json.dump(custom_pattern, f, indent=2)
   ```

## Notes

- All features must be normalized (mean=0, std=1 based on training data)
- Feature names must match exactly (case-sensitive)
- Missing features will be handled by the model's `handle_missing=True` parameter
- Values are typically in range [-3, 3] for normalized data
