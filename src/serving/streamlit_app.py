"""
🛡️ MoE Cybersecurity Detection Dashboard
==========================================
Professional Streamlit UI for network anomaly detection using Mixture-of-Experts model
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path
import time

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="MoE Cybersecurity Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS for Professional Look
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: rgba(255, 255, 255, 0.95);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .attack-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #667eea;
        font-weight: 700;
    }
    h2 {
        color: #764ba2;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Configuration
# ============================================================================
API_URL = "http://localhost:8000"

# Feature definitions for CICIDS dataset
FEATURE_CATEGORIES = {
    "Flow Characteristics": [
        "Flow Duration", "Flow Bytes/s", "Flow Packets/s", 
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"
    ],
    "Forward Packets": [
        "Total Fwd Packets", "Total Length of Fwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Fwd IAT Mean", "Fwd IAT Max", "Fwd IAT Min", "Fwd IAT Std"
    ],
    "Backward Packets": [
        "Total Backward Packets", "Total Length of Bwd Packets",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd IAT Mean", "Bwd IAT Max", "Bwd IAT Min", "Bwd IAT Std"
    ],
    "TCP Flags": [
        "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
        "FIN Flag Count", "RST Flag Count", "URG Flag Count"
    ],
    "Packet Statistics": [
        "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "Min Packet Length", "Max Packet Length"
    ]
}

# Load REAL attack examples from processed CICIDS data
def load_real_examples():
    """Load real attack examples from processed data for accurate predictions"""
    try:
        # Try to load pre-extracted examples
        ddos_path = Path("ddos_example.json")
        normal_path = Path("normal_example.json")
        
        if ddos_path.exists() and normal_path.exists():
            with open(ddos_path, 'r') as f:
                ddos_example = json.load(f)
            with open(normal_path, 'r') as f:
                normal_example = json.load(f)
            return {
                "DDoS Attack (Real)": ddos_example,
                "Normal Traffic (Real)": normal_example
            }
    except Exception as e:
        st.warning(f"Could not load real examples: {e}")
    
    # Fallback to simplified examples
    return {
        "DDoS Attack": {
            "Flow Duration": 0,
            "Total Fwd Packets": 5,
            "Total Backward Packets": 0,
            "SYN Flag Count": 5,
            "ACK Flag Count": 0,
            "Flow Bytes/s": 1000000,
            "Flow Packets/s": 50000
        },
        "Normal Traffic": {
            "Flow Duration": 120000,
            "Total Fwd Packets": 50,
            "Total Backward Packets": 45,
            "SYN Flag Count": 1,
            "ACK Flag Count": 95,
            "Flow Bytes/s": 5000,
            "Flow Packets/s": 15
        }
    }

ATTACK_EXAMPLES = load_real_examples()

# ============================================================================
# Helper Functions
# ============================================================================
# Helper Functions
# ============================================================================

def check_api_status():
    """Check if API is accessible"""
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
    },
    
    "Port Scan": {
        # Basic identifiers  
        " Destination Port": 22,
        
        # Flow characteristics - Single packet probe
        " Flow Duration": 0,
        " Total Fwd Packets": 1,
        " Total Backward Packets": 0,
        "Total Length of Fwd Packets": 60,
        " Total Length of Bwd Packets": 0,
        
        # Forward packet characteristics
        " Fwd Packet Length Max": 60,
        " Fwd Packet Length Min": 60,
        " Fwd Packet Length Mean": 60.0,
        " Fwd Packet Length Std": 0.0,
        
        # Backward packet characteristics (none)
        "Bwd Packet Length Max": 0,
        " Bwd Packet Length Min": 0,
        " Bwd Packet Length Mean": 0.0,
        " Bwd Packet Length Std": 0.0,
        
        # Flow rates
        "Flow Bytes/s": 0.0,
        " Flow Packets/s": 0.0,
        
        # Flow IAT
        " Flow IAT Mean": 0.0,
        " Flow IAT Std": 0.0,
        " Flow IAT Max": 0.0,
        " Flow IAT Min": 0.0,
        
        # Forward IAT
        "Fwd IAT Total": 0,
        " Fwd IAT Mean": 0.0,
        " Fwd IAT Std": 0.0,
        " Fwd IAT Max": 0.0,
        " Fwd IAT Min": 0.0,
        
        # Backward IAT
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
        " Fwd Header Length": 60,
        " Bwd Header Length": 0,
        
        # Packet rates
        "Fwd Packets/s": 0.0,
        " Bwd Packets/s": 0.0,
        
        # Packet length stats
        " Min Packet Length": 60,
        " Max Packet Length": 60,
        " Packet Length Mean": 60.0,
        " Packet Length Std": 0.0,
        " Packet Length Variance": 0.0,
        
        # TCP Flags - Single SYN probe
        "FIN Flag Count": 0,
        " SYN Flag Count": 1,
        " RST Flag Count": 0,
        " PSH Flag Count": 0,
        " ACK Flag Count": 0,
        " URG Flag Count": 0,
        " CWE Flag Count": 0,
        " ECE Flag Count": 0,
        
        # Ratios
        " Down/Up Ratio": 0,
        " Average Packet Size": 60.0,
        " Avg Fwd Segment Size": 60.0,
        " Avg Bwd Segment Size": 0.0,
        
        # Additional headers
        " Fwd Header Length.1": 60,
        
        # Bulk transfer
        "Fwd Avg Bytes/Bulk": 0,
        " Fwd Avg Packets/Bulk": 0,
        " Fwd Avg Bulk Rate": 0,
        " Bwd Avg Bytes/Bulk": 0,
        " Bwd Avg Packets/Bulk": 0,
        "Bwd Avg Bulk Rate": 0,
        
        # Subflows
        "Subflow Fwd Packets": 1,
        " Subflow Fwd Bytes": 60,
        " Subflow Bwd Packets": 0,
        " Subflow Bwd Bytes": 0,
        
        # Window sizes
        "Init_Win_bytes_forward": 5840,
        " Init_Win_bytes_backward": -1,
        
        # Active data packets
        " act_data_pkt_fwd": 0,
        " min_seg_size_forward": 20,
        
        # Active/Idle times
        "Active Mean": 0,
        " Active Std": 0,
        " Active Max": 0,
        " Active Min": 0,
        "Idle Mean": 0,
        " Idle Std": 0,
        " Idle Max": 0,
        " Idle Min": 0
    },
    
    "Web Attack": {
        # Basic identifiers
        " Destination Port": 80,
        
        # Flow characteristics - HTTP exploit attempt
        " Flow Duration": 45000000,
        " Total Fwd Packets": 2100,
        " Total Backward Packets": 1730,
        "Total Length of Fwd Packets": 3200000,
        " Total Length of Bwd Packets": 2450000,
        
        # Forward packet characteristics
        " Fwd Packet Length Max": 1500,
        " Fwd Packet Length Min": 52,
        " Fwd Packet Length Mean": 1523.81,
        " Fwd Packet Length Std": 428.57,
        
        # Backward packet characteristics
        "Bwd Packet Length Max": 1460,
        " Bwd Packet Length Min": 52,
        " Bwd Packet Length Mean": 1416.18,
        " Bwd Packet Length Std": 395.43,
        
        # Flow rates
        "Flow Bytes/s": 125000.0,
        " Flow Packets/s": 85.0,
        
        # Flow IAT
        " Flow IAT Mean": 11764.71,
        " Flow IAT Std": 120000.0,
        " Flow IAT Max": 950000.0,
        " Flow IAT Min": 0.0,
        
        # Forward IAT
        "Fwd IAT Total": 45000000,
        " Fwd IAT Mean": 21428.57,
        " Fwd IAT Std": 135000.0,
        " Fwd IAT Max": 950000.0,
        " Fwd IAT Min": 0.0,
        
        # Backward IAT
        "Bwd IAT Total": 45000000,
        " Bwd IAT Mean": 26011.56,
        " Bwd IAT Std": 145000.0,
        " Bwd IAT Max": 960000.0,
        " Bwd IAT Min": 0.0,
        
        # PSH and URG flags
        "Fwd PSH Flags": 1500,
        " Bwd PSH Flags": 1300,
        " Fwd URG Flags": 0,
        " Bwd URG Flags": 0,
        
        # Header lengths
        " Fwd Header Length": 84000,
        " Bwd Header Length": 69200,
        
        # Packet rates
        "Fwd Packets/s": 46.67,
        " Bwd Packets/s": 38.44,
        
        # Packet length stats
        " Min Packet Length": 52,
        " Max Packet Length": 1500,
        " Packet Length Mean": 1473.76,
        " Packet Length Std": 412.23,
        " Packet Length Variance": 169933.51,
        
        # TCP Flags - KEY: Many PSH (data transfer)
        "FIN Flag Count": 1,
        " SYN Flag Count": 1,
        " RST Flag Count": 0,
        " PSH Flag Count": 2800,  # Lots of data being pushed
        " ACK Flag Count": 3820,
        " URG Flag Count": 0,
        " CWE Flag Count": 0,
        " ECE Flag Count": 0,
        
        # Ratios
        " Down/Up Ratio": 0.77,
        " Average Packet Size": 1473.76,
        " Avg Fwd Segment Size": 1523.81,
        " Avg Bwd Segment Size": 1416.18,
        
        # Additional headers
        " Fwd Header Length.1": 84000,
        
        # Bulk transfer
        "Fwd Avg Bytes/Bulk": 320000,
        " Fwd Avg Packets/Bulk": 210,
        " Fwd Avg Bulk Rate": 7111.11,
        " Bwd Avg Bytes/Bulk": 245000,
        " Bwd Avg Packets/Bulk": 173,
        "Bwd Avg Bulk Rate": 5444.44,
        
        # Subflows
        "Subflow Fwd Packets": 2100,
        " Subflow Fwd Bytes": 3200000,
        " Subflow Bwd Packets": 1730,
        " Subflow Bwd Bytes": 2450000,
        
        # Window sizes
        "Init_Win_bytes_forward": 29200,
        " Init_Win_bytes_backward": 28960,
        
        # Active data packets
        " act_data_pkt_fwd": 2050,
        " min_seg_size_forward": 20,
        
        # Active/Idle times
        "Active Mean": 8000000,
        " Active Std": 2500000,
        " Active Max": 12000000,
        " Active Min": 5000000,
        "Idle Mean": 125000,
        " Idle Std": 85000,
        " Idle Max": 350000,
        " Idle Min": 15000
    },
    
    "Normal Traffic": {
        # Basic identifiers
        " Destination Port": 443,
        
        # Flow characteristics - Regular HTTPS browsing
        " Flow Duration": 120000000,
        " Total Fwd Packets": 950,
        " Total Backward Packets": 850,
        "Total Length of Fwd Packets": 720000,
        " Total Length of Bwd Packets": 780000,
        
        # Forward packet characteristics
        " Fwd Packet Length Max": 1380,
        " Fwd Packet Length Min": 52,
        " Fwd Packet Length Mean": 757.89,
        " Fwd Packet Length Std": 324.67,
        
        # Backward packet characteristics
        "Bwd Packet Length Max": 1460,
        " Bwd Packet Length Min": 52,
        " Bwd Packet Length Mean": 917.65,
        " Bwd Packet Length Std": 387.22,
        
        # Flow rates
        "Flow Bytes/s": 12500.0,
        " Flow Packets/s": 15.0,
        
        # Flow IAT
        " Flow IAT Mean": 66666.67,
        " Flow IAT Std": 85000.0,
        " Flow IAT Max": 750000.0,
        " Flow IAT Min": 5.0,
        
        # Forward IAT
        "Fwd IAT Total": 120000000,
        " Fwd IAT Mean": 126315.79,
        " Fwd IAT Std": 92000.0,
        " Fwd IAT Max": 750000.0,
        " Fwd IAT Min": 10.0,
        
        # Backward IAT
        "Bwd IAT Total": 120000000,
        " Bwd IAT Mean": 141176.47,
        " Bwd IAT Std": 95000.0,
        " Bwd IAT Max": 755000.0,
        " Bwd IAT Min": 10.0,
        
        # PSH and URG flags
        "Fwd PSH Flags": 450,
        " Bwd PSH Flags": 400,
        " Fwd URG Flags": 0,
        " Bwd URG Flags": 0,
        
        # Header lengths
        " Fwd Header Length": 38000,
        " Bwd Header Length": 34000,
        
        # Packet rates
        "Fwd Packets/s": 7.92,
        " Bwd Packets/s": 7.08,
        
        # Packet length stats
        " Min Packet Length": 52,
        " Max Packet Length": 1460,
        " Packet Length Mean": 833.33,
        " Packet Length Std": 356.89,
        " Packet Length Variance": 127370.25,
        
        # TCP Flags - KEY: Balanced, proper handshake
        "FIN Flag Count": 2,
        " SYN Flag Count": 1,  # Single SYN (normal connection)
        " RST Flag Count": 0,
        " PSH Flag Count": 850,
        " ACK Flag Count": 1795,  # Many ACKs (bidirectional)
        " URG Flag Count": 0,
        " CWE Flag Count": 0,
        " ECE Flag Count": 0,
        
        # Ratios
        " Down/Up Ratio": 1.08,
        " Average Packet Size": 833.33,
        " Avg Fwd Segment Size": 757.89,
        " Avg Bwd Segment Size": 917.65,
        
        # Additional headers
        " Fwd Header Length.1": 38000,
        
        # Bulk transfer
        "Fwd Avg Bytes/Bulk": 72000,
        " Fwd Avg Packets/Bulk": 95,
        " Fwd Avg Bulk Rate": 600.0,
        " Bwd Avg Bytes/Bulk": 78000,
        " Bwd Avg Packets/Bulk": 85,
        "Bwd Avg Bulk Rate": 650.0,
        
        # Subflows
        "Subflow Fwd Packets": 950,
        " Subflow Fwd Bytes": 720000,
        " Subflow Bwd Packets": 850,
        " Subflow Bwd Bytes": 780000,
        
        # Window sizes
        "Init_Win_bytes_forward": 64240,
        " Init_Win_bytes_backward": 65535,
        
        # Active data packets
        " act_data_pkt_fwd": 900,
        " min_seg_size_forward": 20,
        
        # Active/Idle times
        "Active Mean": 15000000,
        " Active Std": 5500000,
        " Active Max": 28000000,
        " Active Min": 8000000,
        "Idle Mean": 850000,
        " Idle Std": 325000,
        " Idle Max": 1500000,
        " Idle Min": 125000
    }
# Load REAL attack examples from processed CICIDS data
def load_real_examples():
    """Load real attack examples from processed data for accurate predictions"""
    try:
        # Try to load pre-extracted examples
        ddos_path = Path("ddos_example.json")
        normal_path = Path("normal_example.json")
        
        if ddos_path.exists() and normal_path.exists():
            with open(ddos_path, 'r') as f:
                ddos_example = json.load(f)
            with open(normal_path, 'r') as f:
                normal_example = json.load(f)
            return {
                "DDoS Attack (Real)": ddos_example,
                "Normal Traffic (Real)": normal_example
            }
    except Exception as e:
        st.warning(f"Could not load real examples: {e}")
    
    # Fallback to simplified examples
    return {
        "DDoS Attack": {
            "Flow Duration": 0,
            "Total Fwd Packets": 5,
            "Total Backward Packets": 0,
            "SYN Flag Count": 5,
            "ACK Flag Count": 0,
            "Flow Bytes/s": 1000000,
            "Flow Packets/s": 50000
        },
        "Normal Traffic": {
            "Flow Duration": 120000,
            "Total Fwd Packets": 50,
            "Total Backward Packets": 45,
            "SYN Flag Count": 1,
            "ACK Flag Count": 95,
            "Flow Bytes/s": 5000,
            "Flow Packets/s": 15
        }
    }

ATTACK_EXAMPLES = load_real_examples()

# ============================================================================
# Helper Functions
# ============================================================================

def check_api_status():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model_info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_prediction(features, handle_missing=True):
    """Make a single prediction"""
    try:
        payload = {
            "features": features,
            "handle_missing": handle_missing
        }
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}

def make_batch_prediction(features_list):
    """Make batch predictions"""
    try:
        payload = {
            "features": features_list,
            "handle_missing": True
        }
        response = requests.post(f"{API_URL}/predict_batch", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}

def create_gauge_chart(confidence, prediction):
    """Create a gauge chart for confidence"""
    color = "#00f2fe" if prediction == "Normal" else "#f5576c"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{prediction} Confidence", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f0f2f6'},
                {'range': [50, 80], 'color': '#e0e2e6'},
                {'range': [80, 100], 'color': '#d0d2d6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#667eea", 'family': "Arial"}
    )
    
    return fig

def create_probability_chart(probabilities):
    """Create bar chart for probabilities"""
    df = pd.DataFrame({
        'Class': list(probabilities.keys()),
        'Probability': [v * 100 for v in probabilities.values()]
    })
    
    colors = ['#00f2fe' if c == 'Normal' else '#f5576c' for c in df['Class']]
    
    fig = px.bar(
        df, x='Class', y='Probability',
        color='Class',
        color_discrete_map={'Normal': '#00f2fe', 'Attack': '#f5576c'},
        title="Class Probabilities"
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        yaxis_title="Probability (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    
    return fig

def create_gating_chart(gating_weights):
    """Create pie chart for expert gating weights"""
    if not gating_weights:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=list(gating_weights.keys()),
        values=list(gating_weights.values()),
        hole=.3,
        marker_colors=['#667eea', '#764ba2']
    )])
    
    fig.update_layout(
        title="Expert Contributions",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text='MoE', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

# ============================================================================
# Main Application
# ============================================================================

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>🛡️ MoE Cybersecurity Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Real-time network anomaly detection using Mixture-of-Experts deep learning</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-shield-green.png", width=80)
        st.markdown("### 🔧 Configuration")
        
        # API Status
        api_status = check_api_status()
        if api_status:
            st.success("✅ API Connected")
            model_info = get_model_info()
            if model_info:
                st.info(f"**Model:** {model_info.get('architecture', 'N/A')}")
                st.info(f"**Dataset:** {model_info.get('dataset', 'N/A')}")
                st.info(f"**Features:** {model_info.get('expected_features', 'N/A')}")
        else:
            st.error("❌ API Disconnected")
            st.warning("Start API: `docker-compose up api`")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        # Placeholder for stats
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        total_predictions = len(st.session_state.prediction_history)
        attack_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 'Attack')
        
        st.metric("Total Predictions", total_predictions)
        st.metric("Attacks Detected", attack_count)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard uses a **Mixture-of-Experts** model combining:
        - 🎯 FT-Transformer (Tabular)
        - 📈 1D-CNN (Temporal)
        - 🎭 Gating Network
        """)
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📊 Batch Analysis", "📈 Model Insights", "🔬 Live Monitor"])
    
    # ========================================================================
    # TAB 1: Single Prediction
    # ========================================================================
    with tab1:
        st.markdown("### 🎯 Real-time Threat Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Input Features")
            
            # Initialize session state for features
            if 'current_features' not in st.session_state:
                st.session_state.current_features = {}
            
            # Load example button
            example_type = st.selectbox(
                "Load Example Pattern",
                ["Custom"] + list(ATTACK_EXAMPLES.keys()),
                help="Load pre-configured attack patterns",
                key="example_selector"
            )
            
            # Update features when example changes
            if 'last_example_type' not in st.session_state:
                st.session_state.last_example_type = "Custom"
            
            if example_type != st.session_state.last_example_type:
                if example_type != "Custom":
                    st.session_state.current_features = ATTACK_EXAMPLES[example_type].copy()
                    st.success(f"✨ Loaded {example_type} pattern - values updated below!")
                    st.balloons()  # Visual feedback!
                else:
                    st.session_state.current_features = {}
                st.session_state.last_example_type = example_type
            
            # Show current pattern info
            if example_type != "Custom":
                num_features = len([v for v in st.session_state.current_features.values() if v != 0])
                st.info(f"📋 **{example_type}** pattern loaded with {num_features} non-zero features")
            
            # Feature inputs in expandable sections
            features = {}
            for category, feature_list in FEATURE_CATEGORIES.items():
                with st.expander(f"📁 {category}", expanded=(category == "Flow Characteristics")):
                    cols = st.columns(3)
                    for idx, feature in enumerate(feature_list):
                        with cols[idx % 3]:
                            default_value = st.session_state.current_features.get(feature, 0.0)
                            features[feature] = st.number_input(
                                feature,
                                value=float(default_value),
                                format="%.2f",
                                key=f"single_{feature}_{example_type}"  # Key changes with example
                            )
            
            # Predict button
            if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
                if not api_status:
                    st.error("❌ API is not available. Please start the API server.")
                else:
                    with st.spinner("Analyzing network traffic..."):
                        result = make_prediction(features)
                        
                        if "error" not in result:
                            # Store in history
                            result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.prediction_history.append(result)
                            
                            st.session_state.last_result = result
                        else:
                            st.error(f"❌ Error: {result['error']}")
        
        with col2:
            st.markdown("#### Prediction Results")
            
            if 'last_result' in st.session_state:
                result = st.session_state.last_result
                
                # Prediction badge
                if result['prediction'] == 'Attack':
                    st.markdown(f"""
                    <div class="attack-card">
                        <h2 style="color: white; margin: 0;">⚠️ ATTACK DETECTED</h2>
                        <p style="margin: 5px 0;">Confidence: {result['confidence']*100:.2f}%</p>
                        <p style="margin: 5px 0;">Time: {result['inference_time_ms']:.2f}ms</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <h2 style="color: white; margin: 0;">✅ NORMAL TRAFFIC</h2>
                        <p style="margin: 5px 0;">Confidence: {result['confidence']*100:.2f}%</p>
                        <p style="margin: 5px 0;">Time: {result['inference_time_ms']:.2f}ms</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Gauge chart
                st.plotly_chart(
                    create_gauge_chart(result['confidence'], result['prediction']),
                    use_container_width=True
                )
                
                # Probabilities
                st.plotly_chart(
                    create_probability_chart(result['probabilities']),
                    use_container_width=True
                )
                
                # Gating weights
                if result.get('gating_weights'):
                    st.plotly_chart(
                        create_gating_chart(result['gating_weights']),
                        use_container_width=True
                    )
                
                # Details
                with st.expander("📋 Detailed Results"):
                    st.json(result)
    
    # ========================================================================
    # TAB 2: Batch Analysis
    # ========================================================================
    with tab2:
        st.markdown("### 📊 Batch Traffic Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Upload Traffic Data")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file with network flows",
                type=['csv'],
                help="CSV file with network traffic features"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} flows")
                    
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("🔍 Analyze All Flows", type="primary", use_container_width=True):
                        if not api_status:
                            st.error("❌ API is not available.")
                        else:
                            features_list = df.to_dict('records')
                            
                            with st.spinner(f"Analyzing {len(features_list)} flows..."):
                                batch_result = make_batch_prediction(features_list)
                                
                                if "error" not in batch_result:
                                    st.session_state.batch_result = batch_result
                                    st.session_state.batch_df = df
                                else:
                                    st.error(f"❌ Error: {batch_result['error']}")
                
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
            
            else:
                st.info("💡 Or generate synthetic data:")
                
                num_samples = st.slider("Number of samples", 10, 100, 50)
                attack_ratio = st.slider("Attack ratio", 0.0, 1.0, 0.3, 0.1)
                
                if st.button("🎲 Generate Random Data", use_container_width=True):
                    import numpy as np
                    
                    samples = []
                    for i in range(num_samples):
                        if np.random.random() < attack_ratio:
                            # Generate attack-like pattern
                            sample = ATTACK_EXAMPLES["DDoS"].copy()
                            # Add some randomness
                            for key in sample:
                                sample[key] *= np.random.uniform(0.8, 1.2)
                        else:
                            # Generate normal pattern
                            sample = ATTACK_EXAMPLES["Normal Traffic"].copy()
                            for key in sample:
                                sample[key] *= np.random.uniform(0.9, 1.1)
                        samples.append(sample)
                    
                    with st.spinner(f"Analyzing {num_samples} flows..."):
                        batch_result = make_batch_prediction(samples)
                        
                        if "error" not in batch_result:
                            st.session_state.batch_result = batch_result
                            st.session_state.batch_df = pd.DataFrame(samples)
        
        with col2:
            st.markdown("#### Analysis Results")
            
            if 'batch_result' in st.session_state:
                result = st.session_state.batch_result
                predictions = result['predictions']
                
                # Summary metrics
                total = len(predictions)
                attacks = sum(1 for p in predictions if p['prediction'] == 'Attack')
                normal = total - attacks
                avg_confidence = sum(p['confidence'] for p in predictions) / total
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Total Flows", total)
                with metric_col2:
                    st.metric("Attacks", attacks, delta=f"{attacks/total*100:.1f}%")
                with metric_col3:
                    st.metric("Normal", normal, delta=f"{normal/total*100:.1f}%")
                
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Normal', 'Attack'],
                    values=[normal, attacks],
                    marker_colors=['#00f2fe', '#f5576c'],
                    hole=.4
                )])
                fig.update_layout(
                    title="Traffic Distribution",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                confidences = [p['confidence'] * 100 for p in predictions]
                fig = px.histogram(
                    x=confidences,
                    nbins=20,
                    title="Confidence Distribution",
                    labels={'x': 'Confidence (%)', 'y': 'Count'},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("#### Prediction Details")
                results_df = pd.DataFrame([{
                    'Index': i,
                    'Prediction': p['prediction'],
                    'Confidence': f"{p['confidence']*100:.2f}%",
                    'Inference Time (ms)': f"{p['inference_time_ms']:.2f}"
                } for i, p in enumerate(predictions)])
                
                st.dataframe(results_df, use_container_width=True, height=300)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    # ========================================================================
    # TAB 3: Model Insights
    # ========================================================================
    with tab3:
        st.markdown("### 📈 Model Performance Insights")
        
        if model_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Architecture")
                st.markdown(f"""
                - **Name:** {model_info.get('model_name', 'N/A')}
                - **Architecture:** {model_info.get('architecture', 'N/A')}
                - **Dataset:** {model_info.get('dataset', 'N/A')}
                - **Expected Features:** {model_info.get('expected_features', 'N/A')}
                - **Source:** {model_info.get('source', 'N/A')}
                - **Stage:** {model_info.get('stage', 'N/A')}
                """)
                
                st.markdown("#### Expert Components")
                st.markdown("""
                **1. Tabular Expert (FT-Transformer)**
                - Processes statistical flow features
                - 47 numerical features
                - Embedding dimension: 128
                - Best for: Statistical patterns
                
                **2. Temporal Expert (1D-CNN)**
                - Processes time-series patterns
                - 22 temporal features
                - Multi-scale convolutions
                - Best for: Attack sequences
                
                **3. Gating Network**
                - Dynamically weights expert contributions
                - Temperature-based softmax
                - Adapts to input characteristics
                """)
            
            with col2:
                st.markdown("#### Prediction History")
                
                if st.session_state.prediction_history:
                    history_df = pd.DataFrame([{
                        'Time': p['timestamp'],
                        'Prediction': p['prediction'],
                        'Confidence': f"{p['confidence']*100:.2f}%"
                    } for p in st.session_state.prediction_history[-10:]])
                    
                    st.dataframe(history_df, use_container_width=True)
                    
                    # Visualization
                    pred_counts = pd.Series([p['prediction'] for p in st.session_state.prediction_history]).value_counts()
                    
                    fig = px.bar(
                        x=pred_counts.index,
                        y=pred_counts.values,
                        title="Prediction Distribution",
                        labels={'x': 'Class', 'y': 'Count'},
                        color=pred_counts.index,
                        color_discrete_map={'Normal': '#00f2fe', 'Attack': '#f5576c'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("🗑️ Clear History"):
                        st.session_state.prediction_history = []
                        st.rerun()
                else:
                    st.info("No predictions yet. Try the Single Prediction tab!")
        else:
            st.warning("Model information not available. Check API connection.")
    
    # ========================================================================
    # TAB 4: Live Monitor
    # ========================================================================
    with tab4:
        st.markdown("### 🔬 Live Traffic Monitor")
        
        st.info("🚧 Live monitoring coming soon! This will show real-time traffic analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Features")
            st.markdown("""
            - Real-time packet capture
            - Automatic threat detection
            - Alert notifications
            - Traffic visualization
            - Export to SIEM
            """)
        
        with col2:
            st.markdown("#### Coming Soon")
            st.markdown("""
            - Integration with network interfaces
            - Configurable alert thresholds
            - Attack pattern learning
            - Historical trend analysis
            - Email/Slack notifications
            """)

# ============================================================================
# Run Application
# ============================================================================
if __name__ == "__main__":
    main()
