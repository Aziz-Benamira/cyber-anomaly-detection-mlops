"""
🛡️ MoE Cybersecurity Detection Dashboard
==========================================
Simple Streamlit UI for network anomaly detection using the trained MoE model
"""

import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="MoE Cybersecurity Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ MoE Cybersecurity Detection")
st.markdown("### Network Anomaly Detection using Mixture-of-Experts Model")

API_URL = "http://localhost:8000"

# ============================================================================
# Load Real Examples
# ============================================================================
@st.cache_data
def load_real_examples():
    """Load real attack examples from processed CICIDS data"""
    try:
        ddos_path = Path("ddos_example.json")
        normal_path = Path("normal_example.json")
        
        if ddos_path.exists() and normal_path.exists():
            with open(ddos_path, 'r') as f:
                ddos = json.load(f)
            with open(normal_path, 'r') as f:
                normal = json.load(f)
            return {"DDoS Attack": ddos, "Normal Traffic": normal}
    except Exception as e:
        st.warning(f"Could not load examples: {e}")
    
    return {}

EXAMPLES = load_real_examples()

# ============================================================================
# Helper Functions
# ============================================================================
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def predict(features):
    try:
        r = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ============================================================================
# Main App
# ============================================================================

# Check API status
col1, col2 = st.columns([3, 1])
with col1:
    st.info("📡 Make sure the API is running: `docker-compose -f docker/docker-compose.yml up api`")
with col2:
    if check_api():
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline")
        st.stop()

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Model Info"])

with tab1:
    st.header("Single Traffic Flow Prediction")
    
    # Example selector
    if EXAMPLES:
        col1, col2 = st.columns([2, 1])
        with col1:
            example_choice = st.selectbox(
                "Load Example Pattern",
                ["Custom"] + list(EXAMPLES.keys()),
                key="example_selector"
            )
        
        if example_choice != "Custom" and example_choice in EXAMPLES:
            st.session_state.features = EXAMPLES[example_choice].copy()
            st.success(f"✅ Loaded {example_choice} pattern with {len(EXAMPLES[example_choice])} features")
    
    # Feature input
    st.subheader("Traffic Features")
    
    # Initialize features if not exist
    if 'features' not in st.session_state:
        st.session_state.features = {}
    
    # Show first 10 features for input
    st.markdown("**Key Features** (showing first 10, full pattern sent to API):")
    
    feature_names = list(st.session_state.features.keys())[:10] if st.session_state.features else [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "SYN Flag Count", "ACK Flag Count", "Flow Bytes/s"
    ]
    
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        with cols[i % 2]:
            val = st.session_state.features.get(feat, 0.0) if isinstance(st.session_state.features.get(feat, 0), (int, float)) else 0.0
            st.session_state.features[feat] = st.number_input(
                feat,
                value=float(val),
                format="%.4f",
                key=f"input_{feat}"
            )
    
    if st.session_state.features:
        st.info(f"📦 Total features in pattern: {len(st.session_state.features)}")
    
    # Predict button
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        if not st.session_state.features:
            st.error("Please load an example or enter feature values")
        else:
            with st.spinner("Analyzing traffic pattern..."):
                result = predict(st.session_state.features)
                
                if result:
                    prediction = result['prediction']
                    confidence = result['confidence'] * 100
                    
                    # Display result
                    if prediction == "Attack":
                        st.error(f"🚨 **ATTACK DETECTED**")
                        st.metric("Confidence", f"{confidence:.2f}%")
                    else:
                        st.success(f"✅ **NORMAL TRAFFIC**")
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Show details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Probabilities")
                        st.json(result['probabilities'])
                    
                    with col2:
                        st.subheader("Expert Weights")
                        st.json(result['gating_weights'])

with tab2:
    st.header("Model Information")
    
    try:
        r = requests.get(f"{API_URL}/model_info", timeout=2)
        if r.status_code == 200:
            info = r.json()
            st.json(info)
        else:
            st.error("Could not fetch model info")
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.markdown("---")
    st.subheader("Training Performance")
    st.markdown("""
    **MoE Model - CICIDS Dataset**
    - F1-Score: 98.35%
    - Precision: 97.16%
    - Recall: 99.56%
    - AUC-PR: 99.91%
    
    **Architecture:**
    - Tabular Expert (FT-Transformer): 47 features
    - Temporal Expert (1D-CNN): 25 features
    - Total Parameters: 733,237
    """)

st.markdown("---")
st.caption("🛡️ MoE Cybersecurity Detection Dashboard | Powered by PyTorch & FastAPI")
