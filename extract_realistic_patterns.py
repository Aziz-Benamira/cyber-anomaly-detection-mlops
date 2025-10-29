"""Extract realistic attack patterns from CICIDS dataset"""
import pandas as pd
import json

# Load DDoS data
print("=" * 70)
print("📊 Extracting REALISTIC Attack Patterns from CICIDS Dataset")
print("=" * 70)

print("\n🔴 Loading DDoS data...")
ddos_df = pd.read_csv('data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
print(f"Total DDoS samples: {len(ddos_df):,}")

# Get a typical DDoS sample (median of first 100 rows)
ddos_sample = ddos_df.head(100).median(numeric_only=True)

print("\n🟢 Loading Normal data (Monday)...")
normal_df = pd.read_csv('data/raw/Monday-WorkingHours.pcap_ISCX.csv')
# Filter for actual normal traffic (not attack)
normal_df = normal_df[normal_df[' Label'] == 'BENIGN']
print(f"Total Normal samples: {len(normal_df):,}")

normal_sample = normal_df.head(100).median(numeric_only=True)

print("\n" + "=" * 70)
print("📋 DDoS Pattern (Median of 100 samples):")
print("=" * 70)
for col in [' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 
            ' SYN Flag Count', ' ACK Flag Count', 'Flow Bytes/s', ' Flow Packets/s']:
    if col in ddos_sample.index:
        print(f"{col:30s}: {ddos_sample[col]:,.2f}")

print("\n" + "=" * 70)
print("📋 Normal Pattern (Median of 100 samples):")
print("=" * 70)
for col in [' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
            ' SYN Flag Count', ' ACK Flag Count', 'Flow Bytes/s', ' Flow Packets/s']:
    if col in normal_sample.index:
        print(f"{col:30s}: {normal_sample[col]:,.2f}")

# Create complete patterns as dictionaries
print("\n" + "=" * 70)
print("✅ Creating Complete 78-Feature Patterns")
print("=" * 70)

# Convert to dictionaries with proper feature names
ddos_dict = {col: float(ddos_sample[col]) if col in ddos_sample else 0.0 
             for col in ddos_df.columns if col != ' Label'}

normal_dict = {col: float(normal_sample[col]) if col in normal_sample else 0.0
               for col in normal_df.columns if col != ' Label'}

print(f"\nDDoS pattern features: {len(ddos_dict)}")
print(f"Normal pattern features: {len(normal_dict)}")

# Save patterns
with open('realistic_ddos_pattern.json', 'w') as f:
    json.dump(ddos_dict, f, indent=2)

with open('realistic_normal_pattern.json', 'w') as f:
    json.dump(normal_dict, f, indent=2)

print("\n✅ Saved realistic_ddos_pattern.json and realistic_normal_pattern.json")
print("\nKey DDoS indicators:")
print(f"  SYN Flags: {ddos_dict.get(' SYN Flag Count', 0):.0f}")
print(f"  ACK Flags: {ddos_dict.get(' ACK Flag Count', 0):.0f}")
print(f"  Fwd Packets: {ddos_dict.get(' Total Fwd Packets', 0):.0f}")
print(f"  Bwd Packets: {ddos_dict.get(' Total Backward Packets', 0):.0f}")
