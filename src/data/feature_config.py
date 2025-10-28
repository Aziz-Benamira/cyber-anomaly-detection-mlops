"""
Feature Split Configuration for MoE

Defines which features belong to Tabular Expert vs Temporal Expert for each dataset.

CICIDS (72 features after preprocessing):
- Temporal features (22): Timing and flow rate statistics
  * Flow Duration (1)
  * Flow Bytes/s, Flow Packets/s (14, 15)
  * Flow IAT: Mean, Std, Max, Min (16-19)
  * Fwd IAT: Total, Mean, Std, Max, Min (20-24)
  * Bwd IAT: Total, Mean, Std, Max, Min (25-29)
  * Active: Mean, Std, Max, Min (70-73)
  * Idle: Mean, Std, Max, Min (74-77)

- Tabular features (50): All other features
  * Packet counts, lengths, flags, protocols, etc.

UNSW (39 numerical features after preprocessing):
- Temporal features (8): Duration and jitter statistics
  * dur: Record total duration (1)
  * sjit: Source jitter (18)
  * djit: Destination jitter (19)
  * sinpkt: Source inter-packet arrival time (16)
  * dinpkt: Destination inter-packet arrival time (17)
  * tcprtt: TCP connection setup round-trip time (24)
  * synack: TCP connection setup time (25)
  * ackdat: TCP connection setup acknowledgment data (26)

- Tabular features (31): All other numerical features
  * Packet counts, bytes, TTL, load, loss, window, connection counts, etc.
"""

# CICIDS Feature Names (in order after preprocessing, excluding Label)
CICIDS_FEATURES = [
    ' Destination Port',
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Backward Packets',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max',
    ' Fwd Packet Length Min',
    ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std',
    'Bwd Packet Length Max',
    ' Bwd Packet Length Min',
    ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std',
    'Flow Bytes/s',
    ' Flow Packets/s',
    ' Flow IAT Mean',
    ' Flow IAT Std',
    ' Flow IAT Max',
    ' Flow IAT Min',
    'Fwd IAT Total',
    ' Fwd IAT Mean',
    ' Fwd IAT Std',
    ' Fwd IAT Max',
    ' Fwd IAT Min',
    'Bwd IAT Total',
    ' Bwd IAT Mean',
    ' Bwd IAT Std',
    ' Bwd IAT Max',
    ' Bwd IAT Min',
    'Fwd PSH Flags',
    ' Bwd PSH Flags',
    ' Fwd URG Flags',
    ' Bwd URG Flags',
    ' Fwd Header Length',
    ' Bwd Header Length',
    'Fwd Packets/s',
    ' Bwd Packets/s',
    ' Min Packet Length',
    ' Max Packet Length',
    ' Packet Length Mean',
    ' Packet Length Std',
    ' Packet Length Variance',
    'FIN Flag Count',
    ' SYN Flag Count',
    ' RST Flag Count',
    ' PSH Flag Count',
    ' ACK Flag Count',
    ' URG Flag Count',
    ' CWE Flag Count',
    ' ECE Flag Count',
    ' Down/Up Ratio',
    ' Average Packet Size',
    ' Avg Fwd Segment Size',
    ' Avg Bwd Segment Size',
    ' Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk',
    ' Fwd Avg Packets/Bulk',
    ' Fwd Avg Bulk Rate',
    ' Bwd Avg Bytes/Bulk',
    ' Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets',
    ' Subflow Fwd Bytes',
    ' Subflow Bwd Packets',
    ' Subflow Bwd Bytes',
    'Init_Win_bytes_forward',
    ' Init_Win_bytes_backward',
    ' act_data_pkt_fwd',
    ' min_seg_size_forward',
    'Active Mean',
    ' Active Std',
    ' Active Max',
    ' Active Min',
    'Idle Mean',
    ' Idle Std',
    ' Idle Max',
    ' Idle Min',
]

# CICIDS Temporal Feature Indices (0-indexed after preprocessing, Label removed)
# After preprocessing: 72 features (78 original - 6 removed duplicates/corrupted columns)
# Temporal features based on feature names
CICIDS_TEMPORAL_INDICES = [
    1,   # Flow Duration
    14,  # Flow Bytes/s
    15,  # Flow Packets/s
    16,  # Flow IAT Mean
    17,  # Flow IAT Std
    18,  # Flow IAT Max
    19,  # Flow IAT Min
    20,  # Fwd IAT Total
    21,  # Fwd IAT Mean
    22,  # Fwd IAT Std
    23,  # Fwd IAT Max
    24,  # Fwd IAT Min
    25,  # Bwd IAT Total
    26,  # Bwd IAT Mean
    27,  # Bwd IAT Std
    28,  # Bwd IAT Max
    29,  # Bwd IAT Min
    # Active/Idle are at the end (indices 64-71 in the 72-feature array)
    64,  # Active Mean
    65,  # Active Std
    66,  # Active Max
    67,  # Active Min
    68,  # Idle Mean
    69,  # Idle Std
    70,  # Idle Max
    71,  # Idle Min
]

# CICIDS Tabular Feature Indices (all others)
CICIDS_TABULAR_INDICES = [i for i in range(78) if i not in CICIDS_TEMPORAL_INDICES and i != 78]  # Exclude Label

# UNSW Feature Names (numerical only, in order after preprocessing)
UNSW_NUMERICAL_FEATURES = [
    'id',
    'dur',
    'spkts',
    'dpkts',
    'sbytes',
    'dbytes',
    'rate',
    'sttl',
    'dttl',
    'sload',
    'dload',
    'sloss',
    'dloss',
    'sinpkt',
    'dinpkt',
    'sjit',
    'djit',
    'swin',
    'stcpb',
    'dtcpb',
    'dwin',
    'tcprtt',
    'synack',
    'ackdat',
    'smean',
    'dmean',
    'trans_depth',
    'response_body_len',
    'ct_srv_src',
    'ct_state_ttl',
    'ct_dst_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm',
    'is_ftp_login',
    'ct_ftp_cmd',
    'ct_flw_http_mthd',
    'ct_src_ltm',
    'ct_srv_dst',
]

# UNSW Categorical Features
UNSW_CATEGORICAL_FEATURES = ['proto', 'service', 'state']

# UNSW Temporal Feature Indices (0-indexed in NUMERICAL features only, after categorical encoding)
# After preprocessing, categoricals are one-hot encoded, so indices shift
# Assuming preprocessing: [numerical features] + [one-hot encoded categoricals]
# Temporal features are in the numerical part
UNSW_TEMPORAL_INDICES = [
    1,   # dur
    13,  # sinpkt
    14,  # dinpkt
    15,  # sjit
    16,  # djit
    21,  # tcprtt
    22,  # synack
    23,  # ackdat
]

# UNSW Tabular Feature Indices (all numerical except temporal, plus categoricals)
# Will be calculated dynamically in preprocessing based on final feature count


def get_feature_split(dataset: str, total_features: int):
    """
    Get feature indices for tabular and temporal experts.
    
    Args:
        dataset: 'CICIDS' or 'UNSW'
        total_features: Total number of features after preprocessing
    
    Returns:
        dict with 'tabular_indices' and 'temporal_indices' lists
    """
    dataset = dataset.upper()
    
    if dataset == 'CICIDS':
        temporal_indices = CICIDS_TEMPORAL_INDICES
        tabular_indices = [i for i in range(total_features) if i not in temporal_indices]
        
        # Feature names for first 22 temporal features
        temporal_feature_names = [
            'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        
        return {
            'tabular_indices': tabular_indices,
            'temporal_indices': temporal_indices,
            'n_tabular': len(tabular_indices),
            'n_temporal': len(temporal_indices),
            'temporal_names': temporal_feature_names
        }
    
    elif dataset == 'UNSW':
        # For UNSW, temporal features are in the numerical part
        # After one-hot encoding, the indices might shift
        # Temporal indices are based on position in final feature matrix
        temporal_indices = UNSW_TEMPORAL_INDICES
        tabular_indices = [i for i in range(total_features) if i not in temporal_indices]
        
        return {
            'tabular_indices': tabular_indices,
            'temporal_indices': temporal_indices,
            'n_tabular': len(tabular_indices),
            'n_temporal': len(temporal_indices),
            'temporal_names': [UNSW_NUMERICAL_FEATURES[i] for i in temporal_indices if i < len(UNSW_NUMERICAL_FEATURES)]
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    """Test feature split configuration."""
    
    print("=" * 80)
    print("FEATURE SPLIT CONFIGURATION TEST")
    print("=" * 80)
    
    # Test CICIDS
    print("\n### CICIDS ###")
    cicids_split = get_feature_split('CICIDS', total_features=72)
    print(f"Total features: 72")
    print(f"Tabular features: {cicids_split['n_tabular']}")
    print(f"Temporal features: {cicids_split['n_temporal']}")
    print(f"\nTemporal feature indices: {cicids_split['temporal_indices']}")
    print(f"\nTemporal feature names:")
    for name in cicids_split['temporal_names']:
        print(f"  - {name}")
    
    # Test UNSW
    print("\n" + "=" * 80)
    print("\n### UNSW ###")
    unsw_split = get_feature_split('UNSW', total_features=194)  # After one-hot encoding
    print(f"Total features: 194 (after one-hot encoding)")
    print(f"Tabular features: {unsw_split['n_tabular']}")
    print(f"Temporal features: {unsw_split['n_temporal']}")
    print(f"\nTemporal feature indices: {unsw_split['temporal_indices']}")
    print(f"\nTemporal feature names:")
    for name in unsw_split['temporal_names']:
        print(f"  - {name}")
    
    print("\n" + "=" * 80)
    print("✅ FEATURE SPLIT CONFIGURED")
    print("=" * 80)
