"""Debug script to see what features are being sent to the API"""
import requests
import json

# Load the DDoS pattern from streamlit
DDOS_PATTERN = {
    # Basic identifiers
    " Destination Port": 80,
    
    # Flow characteristics - High volume, sustained attack
    " Flow Duration": 12000000,
    " Total Fwd Packets": 180000,
    " Total Backward Packets": 0,
    "Total Length of Fwd Packets": 7200000,
    " Total Length of Bwd Packets": 0,
    
    # Forward packet characteristics
    " Fwd Packet Length Max": 60,
    " Fwd Packet Length Min": 40,
    " Fwd Packet Length Mean": 40.0,
    " Fwd Packet Length Std": 5.0,
    
    # Backward packet characteristics (all zero - one-way attack)
    "Bwd Packet Length Max": 0,
    " Bwd Packet Length Min": 0,
    " Bwd Packet Length Mean": 0.0,
    " Bwd Packet Length Std": 0.0,
    
    # Flow rates - Very high
    "Flow Bytes/s": 600.0,
    " Flow Packets/s": 15.0,
    
    # Flow IAT
    " Flow IAT Mean": 66.67,
    " Flow IAT Std": 450.0,
    " Flow IAT Max": 5000.0,
    " Flow IAT Min": 0.0,
    
    # Forward IAT
    "Fwd IAT Total": 12000000,
    " Fwd IAT Mean": 66.67,
    " Fwd IAT Std": 450.0,
    " Fwd IAT Max": 5000.0,
    " Fwd IAT Min": 0.0,
    
    # Backward IAT (all zero)
    "Bwd IAT Total": 0,
    " Bwd IAT Mean": 0.0,
    " Bwd IAT Std": 0.0,
    " Bwd IAT Max": 0.0,
    " Bwd IAT Min": 0.0,
    
    # PSH and URG flags
    "Fwd PSH Flags": 0,
    " Bwd PSH Flags": 0,
    " Fwd URG Flags": 0,
    " Bwd URG Flags": 0,
    
    # Header lengths
    " Fwd Header Length": 7200000,
    " Bwd Header Length": 0,
    
    # Packet rates
    "Fwd Packets/s": 15.0,
    " Bwd Packets/s": 0.0,
    
    # Packet length stats
    " Min Packet Length": 40,
    " Max Packet Length": 60,
    " Packet Length Mean": 40.0,
    " Packet Length Std": 5.0,
    " Packet Length Variance": 25.0,
    
    # TCP Flags - KEY: Massive SYN flood
    "FIN Flag Count": 0,
    " SYN Flag Count": 180000,  # HUGE number of SYNs
    " RST Flag Count": 0,
    " PSH Flag Count": 0,
    " ACK Flag Count": 0,  # NO ACKs
    " URG Flag Count": 0,
    " CWE Flag Count": 0,
    " ECE Flag Count": 0,
    
    # Ratios
    " Down/Up Ratio": 0,
    " Average Packet Size": 40.0,
    " Avg Fwd Segment Size": 40.0,
    " Avg Bwd Segment Size": 0.0,
    
    # Additional headers
    " Fwd Header Length.1": 40,
    
    # Bulk transfer (none)
    "Fwd Avg Bytes/Bulk": 0,
    " Fwd Avg Packets/Bulk": 0,
    " Fwd Avg Bulk Rate": 0,
    " Bwd Avg Bytes/Bulk": 0,
    " Bwd Avg Packets/Bulk": 0,
    "Bwd Avg Bulk Rate": 0,
    
    # Subflows
    "Subflow Fwd Packets": 180000,
    " Subflow Fwd Bytes": 7200000,
    " Subflow Bwd Packets": 0,
    " Subflow Bwd Bytes": 0,
    
    # Window sizes
    "Init_Win_bytes_forward": 8192,
    " Init_Win_bytes_backward": -1,
    
    # Active data packets
    " act_data_pkt_fwd": 0,
    " min_seg_size_forward": 40,
    
    # Active/Idle times
    "Active Mean": 5000000,
    " Active Std": 0,
    " Active Max": 5000000,
    " Active Min": 5000000,
    "Idle Mean": 0,
    " Idle Std": 0,
    " Idle Max": 0,
    " Idle Min": 0
}

print("=" * 70)
print("📊 DDoS Pattern Feature Analysis")
print("=" * 70)
print(f"\nTotal features in pattern: {len(DDOS_PATTERN)}")
print(f"\nPattern JSON:")
print(json.dumps(DDOS_PATTERN, indent=2))

print("\n" + "=" * 70)
print("🔍 Key Attack Indicators:")
print("=" * 70)
print(f"  SYN Flag Count: {DDOS_PATTERN[' SYN Flag Count']:,}")
print(f"  ACK Flag Count: {DDOS_PATTERN[' ACK Flag Count']}")
print(f"  Total Fwd Packets: {DDOS_PATTERN[' Total Fwd Packets']:,}")
print(f"  Total Bwd Packets: {DDOS_PATTERN[' Total Backward Packets']}")
print(f"  Down/Up Ratio: {DDOS_PATTERN[' Down/Up Ratio']}")

print("\n" + "=" * 70)
print("🚀 Sending to API...")
print("=" * 70)

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": DDOS_PATTERN}
)

if response.status_code == 200:
    result = response.json()
    print("\n✅ API Response:")
    print(json.dumps(result, indent=2))
else:
    print(f"\n❌ API Error: {response.status_code}")
    print(response.text)
