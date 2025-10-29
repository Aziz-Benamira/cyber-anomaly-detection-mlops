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

# ============================================================================
# Load Real Examples
# ============================================================================
@st.cache_data
def load_all_examples():
    """Load all attack examples from JSON files"""
    examples = {}
    example_files = {
        "DDoS Attack": "ddos_attack.json",
        "Port Scan": "port_scan.json", 
        "Web Attack": "web_attack.json",
        "Brute Force": "brute_force.json",
        "Normal Traffic": "normal_traffic.json",
        "Normal HTTPS": "normal_https.json"
    }
    
    for name, filename in example_files.items():
        filepath = Path(filename)
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    examples[name] = json.load(f)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    
    return examples

ATTACK_EXAMPLES = load_all_examples()

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
        response = requests.get(f"{API_URL}/model_info", timeout=2)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def make_prediction(features):
    """Make a prediction using the API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

# ============================================================================
# Visualization Functions
# ============================================================================

def create_gauge_chart(confidence, prediction):
    """Create a gauge chart for confidence"""
    color = "#f5576c" if prediction == "Attack" else "#4facfe"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{prediction} Confidence", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#e0e0e0'},
                {'range': [50, 75], 'color': '#c0c0c0'},
                {'range': [75, 100], 'color': '#a0a0a0'}
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
    """Create a bar chart for class probabilities"""
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]
    colors = ['#f5576c' if c == 'Attack' else '#4facfe' for c in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker_color=colors,
            text=[f'{p:.2f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#667eea"}
    )
    
    return fig

def create_gating_chart(gating_weights):
    """Create a pie chart for expert gating weights"""
    experts = list(gating_weights.keys())
    weights = [gating_weights[e] * 100 for e in experts]
    
    fig = go.Figure(data=[go.Pie(
        labels=experts,
        values=weights,
        hole=0.4,
        marker=dict(colors=['#667eea', '#764ba2']),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Expert Contribution",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#667eea"}
    )
    
    return fig

# ============================================================================
# Main Application
# ============================================================================

def main():
    # Header
    st.title("🛡️ MoE Cybersecurity Detection")
    st.markdown("### Advanced Network Anomaly Detection using Mixture-of-Experts")
    
    # API Status Check
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info("📡 Ensure API is running: `docker-compose -f docker/docker-compose.yml up api`")
    with col2:
        if check_api_status():
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            st.stop()
    with col3:
        model_info = get_model_info()
        if model_info:
            st.success("✅ Model Loaded")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Prediction",
        "📊 Batch Analysis", 
        "📈 Model Insights",
        "⚙️ Live Monitor"
    ])
    
    # ========================================================================
    # TAB 1: Single Prediction
    # ========================================================================
    with tab1:
        st.header("Single Traffic Flow Analysis")
        
        # Initialize session state
        if 'current_features' not in st.session_state:
            if "Normal Traffic" in ATTACK_EXAMPLES:
                st.session_state.current_features = ATTACK_EXAMPLES["Normal Traffic"].copy()
                st.session_state.last_example = "Normal Traffic"
            else:
                st.session_state.current_features = {}
                st.session_state.last_example = "Custom"
        
        if 'last_example' not in st.session_state:
            st.session_state.last_example = "Normal Traffic"
        
        # Example selector
        example_type = st.selectbox(
            "📋 Load Example Pattern",
            ["Custom"] + list(ATTACK_EXAMPLES.keys()),
            key="example_selector"
        )
        
        # Auto-load when selection changes
        if example_type != st.session_state.last_example:
            if example_type == "Custom":
                # Switching to Custom - initialize with zeros or keep current
                if not st.session_state.current_features or st.session_state.last_example != "Custom":
                    if ATTACK_EXAMPLES:
                        example_features = list(next(iter(ATTACK_EXAMPLES.values())).keys())
                        st.session_state.current_features = {f: 0.0 for f in example_features}
                st.session_state.last_example = "Custom"
                st.rerun()
            elif example_type in ATTACK_EXAMPLES:
                st.session_state.current_features = ATTACK_EXAMPLES[example_type].copy()
                st.session_state.last_example = example_type
                st.success(f"✅ Loaded {example_type}!")
                st.rerun()
        
        # Display current pattern info
        if st.session_state.current_features:
            st.info(f"📦 **Current Pattern**: {st.session_state.last_example} | **Features**: {len(st.session_state.current_features)}")
        
        st.markdown("---")
        
        # Feature inputs
        if st.session_state.last_example == "Custom":
            # Custom mode - show ALL 72 features in expandable sections
            st.subheader("Custom Traffic Features (All 72 Features)")
            st.markdown("*Edit any feature value to create custom traffic patterns*")
            
            # Initialize with zeros if empty
            if not st.session_state.current_features:
                # Load all feature names from an example
                if ATTACK_EXAMPLES:
                    example_features = list(next(iter(ATTACK_EXAMPLES.values())).keys())
                    st.session_state.current_features = {f: 0.0 for f in example_features}
            
            all_features = list(st.session_state.current_features.keys())
            
            # Organize features into categories
            tab_features = st.tabs(["📊 Flow Features (0-23)", "🔄 Forward Features (24-47)", "⬅️ Backward Features (48-71)"])
            
            # CRITICAL FIX: Read values from number_input and store separately, then update session_state
            temp_values = {}
            
            with tab_features[0]:
                features_chunk = all_features[0:24]
                cols = st.columns(3)
                for i, feat in enumerate(features_chunk):
                    with cols[i % 3]:
                        val = st.session_state.current_features[feat]
                        temp_values[feat] = st.number_input(
                            feat,
                            value=float(val),
                            format="%.6f",
                            key=f"custom_{feat}"
                        )
            
            with tab_features[1]:
                features_chunk = all_features[24:48]
                cols = st.columns(3)
                for i, feat in enumerate(features_chunk):
                    with cols[i % 3]:
                        val = st.session_state.current_features[feat]
                        temp_values[feat] = st.number_input(
                            feat,
                            value=float(val),
                            format="%.6f",
                            key=f"custom_{feat}"
                        )
            
            with tab_features[2]:
                features_chunk = all_features[48:72]
                cols = st.columns(3)
                for i, feat in enumerate(features_chunk):
                    with cols[i % 3]:
                        val = st.session_state.current_features[feat]
                        temp_values[feat] = st.number_input(
                            feat,
                            value=float(val),
                            format="%.6f",
                            key=f"custom_{feat}"
                        )
            
            # Update session_state with all the values from number_inputs
            st.session_state.current_features.update(temp_values)
        
        else:
            # Example mode - show key features only
            st.subheader("Key Traffic Features (Preview)")
            st.markdown("*Showing 12 important features. Full 72-feature pattern is loaded and will be sent to API.*")
            
            # Get first 12 features from current pattern
            if st.session_state.current_features:
                key_features = list(st.session_state.current_features.keys())[:12]
            else:
                key_features = []
            
            cols = st.columns(3)
            for i, feat in enumerate(key_features):
                if feat in st.session_state.current_features:
                    with cols[i % 3]:
                        val = st.session_state.current_features[feat]
                        st.session_state.current_features[feat] = st.number_input(
                            feat,
                            value=float(val),
                            format="%.6f",
                            key=f"feat_{feat}_{st.session_state.last_example}"
                        )
            
            # Expandable section to show/edit ALL features
            with st.expander("🔧 Advanced: Edit All 72 Features"):
                st.markdown("*Edit any feature in the complete pattern*")
                all_features = list(st.session_state.current_features.keys())
                
                # Split into 3 columns
                col1, col2, col3 = st.columns(3)
                third = len(all_features) // 3
                
                with col1:
                    for feat in all_features[0:third]:
                        val = st.session_state.current_features[feat]
                        st.session_state.current_features[feat] = st.number_input(
                            feat,
                            value=float(val),
                            format="%.6f",
                            key=f"all_{feat}_{st.session_state.last_example}_1"
                        )
                
                with col2:
                    for feat in all_features[third:2*third]:
                        val = st.session_state.current_features[feat]
                        st.session_state.current_features[feat] = st.number_input(
                            feat,
                            value=float(val),
                            format="%.6f",
                            key=f"all_{feat}_{st.session_state.last_example}_2"
                        )
                
                with col3:
                    for feat in all_features[2*third:]:
                        val = st.session_state.current_features[feat]
                        st.session_state.current_features[feat] = st.number_input(
                            feat,
                            value=float(val),
                            format="%.6f",
                            key=f"all_{feat}_{st.session_state.last_example}_3"
                        )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Analyze Traffic", type="primary", use_container_width=True, key="predict_btn"):
            if not st.session_state.current_features:
                st.error("⚠️ Please load an example pattern first!")
            else:
                with st.spinner("🔍 Analyzing traffic pattern..."):
                    result = make_prediction(st.session_state.current_features)
                    
                    if result:
                        prediction = result['prediction']
                        confidence = result['confidence']
                        
                        # Display results with visualizations
                        st.markdown("### 🎯 Analysis Results")
                        
                        # Alert banner
                        if prediction == "Attack":
                            st.error(f"### 🚨 **ATTACK DETECTED** 🚨")
                        else:
                            st.success(f"### ✅ **NORMAL TRAFFIC** ✅")
                        
                        # Visualizations in 3 columns
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Gauge chart
                            gauge_fig = create_gauge_chart(confidence, prediction)
                            st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        with col2:
                            # Probability chart
                            prob_fig = create_probability_chart(result['probabilities'])
                            st.plotly_chart(prob_fig, use_container_width=True)
                        
                        with col3:
                            # Gating weights chart
                            gating_fig = create_gating_chart(result['gating_weights'])
                            st.plotly_chart(gating_fig, use_container_width=True)
                        
                        # Detailed metrics
                        st.markdown("---")
                        st.subheader("📊 Detailed Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Prediction", prediction, delta=None)
                        
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        
                        with col3:
                            tabular_weight = result['gating_weights']['Tabular Expert'] * 100
                            st.metric("Tabular Expert", f"{tabular_weight:.1f}%")
                        
                        with col4:
                            temporal_weight = result['gating_weights']['Temporal Expert'] * 100
                            st.metric("Temporal Expert", f"{temporal_weight:.1f}%")
                        
                        # Raw data expander
                        with st.expander("🔍 View Raw Response"):
                            st.json(result)
    
    # ========================================================================
    # TAB 2: Batch Analysis
    # ========================================================================
    with tab2:
        st.header("Batch Traffic Analysis")
        st.info("📝 Analyze multiple traffic flows from CSV file")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"📊 Loaded {len(df)} samples")
            st.dataframe(df.head())
            
            if st.button("🚀 Analyze All Samples", type="primary"):
                st.warning("Batch prediction not yet implemented in API")
        else:
            st.markdown("**Sample CSV available**: `data/sample_traffic.csv`")
    
    # ========================================================================
    # TAB 3: Model Insights
    # ========================================================================
    with tab3:
        st.header("Model Architecture & Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Architecture")
            st.markdown("""
            **Mixture-of-Experts (MoE) Model**
            
            - **Tabular Expert** (FT-Transformer)
              - Input: 47 statistical features
              - Layers: 3 transformer layers
              - Attention heads: 8
              - Embedding dim: 128
            
            - **Temporal Expert** (1D-CNN)
              - Input: 25 temporal features  
              - Conv layers: 32 → 64 → 64 channels
              - Pooling: Adaptive average
              - Output dim: 128
            
            - **Gating Network** (Dense)
              - Dynamic expert weighting
              - Temperature: 1.0
            
            - **Classifier**
              - Hidden: 256 dims
              - Dropout: 0.1
              - Output: 2 classes
            
            **Total Parameters**: 733,237
            """)
        
        with col2:
            st.subheader("📈 Performance Metrics")
            st.markdown("""
            **Training Results (CICIDS Dataset)**
            
            | Metric | Value |
            |--------|-------|
            | **F1-Score** | 98.35% |
            | **Precision** | 97.16% |
            | **Recall** | 99.56% |
            | **AUC-PR** | 99.91% |
            | **Accuracy** | 98.88% |
            
            **Confusion Matrix (Validation)**
            - True Negatives: 36,416 (98.96%)
            - False Positives: 384 (1.04%)
            - False Negatives: 58 (0.44%)
            - True Positives: 13,142 (99.56%)
            
            **Key Achievements:**
            - ✅ Catches 99.56% of attacks
            - ✅ Only 1.04% false alarm rate
            - ✅ Minimal missed attacks (58/13,200)
            """)
        
        st.markdown("---")
        
        # Model info from API
        if model_info:
            st.subheader("🔧 Current Model Status")
            st.json(model_info)
    
    # ========================================================================
    # TAB 4: Live Monitor
    # ========================================================================
    with tab4:
        st.header("Live System Monitor")
        st.info("⚡ Real-time monitoring dashboard (Phase 4: Prometheus + Grafana integration)")
        
        st.markdown("""
        **Coming Soon:**
        - Real-time traffic analysis
        - Attack frequency charts
        - Expert utilization graphs
        - System performance metrics
        - Alert history
        
        **Status**: Awaiting Prometheus/Grafana setup (Phase 4)
        """)
    
    # Footer
    st.markdown("---")
    st.caption("🛡️ MoE Cybersecurity Detection Dashboard | Powered by PyTorch, FastAPI & Streamlit")

if __name__ == "__main__":
    main()
