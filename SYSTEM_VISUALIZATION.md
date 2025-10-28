# 🎨 MoE System Visualization

## Complete Pipeline: Training → Inference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw PCAP Files
         │
         ├─ Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
         ├─ Monday-WorkingHours.pcap_ISCX.csv
         └─ ...
         │
         ↓
    ┌────────────────┐
    │  Preprocessing │  ← src/data/preprocess.py
    └───────┬────────┘
            │
            ├─ Feature Engineering (72 features for CICIDS)
            ├─ Feature Splitting (47 tabular + 25 temporal)
            ├─ Normalization (StandardScaler)
            └─ Class Balancing (WeightedRandomSampler)
            │
            ↓
    ┌────────────────────────────────────────────┐
    │         STAGE 1: PRETRAIN TABULAR          │
    │                                            │
    │  Input: Tabular features only (47)        │
    │  Model: FT-Transformer (~648k params)     │
    │  Loss: Masked Feature Modeling (MFM)      │
    │  Output: cicids_tabular_pretrained.pt     │
    │  Result: MFM Loss = 0.6526 (5 minutes)    │
    └────────────────┬───────────────────────────┘
                     │
                     ↓
    ┌────────────────────────────────────────────┐
    │         STAGE 2: TRAIN FULL MOE            │
    │                                            │
    │  Input: All features (47 tabular + 25 temp)│
    │  Model: MoE (733k params total)           │
    │    ├─ Tabular Expert (pretrained)         │
    │    ├─ Temporal Expert (1D CNN, 27k)       │
    │    ├─ Gating Network (42k)                │
    │    └─ Classifier (MLP)                    │
    │  Loss: Weighted CrossEntropy              │
    │  Output: cicids_moe_best.pt               │
    │  Result: F1 = 98.3% (10 epochs)           │
    └────────────────┬───────────────────────────┘
                     │
                     ↓
    ┌────────────────────────────────────────────┐
    │      SAVED MODEL CHECKPOINT                │
    │                                            │
    │  models/weights/cicids_moe_best.pt        │
    │    - model_state_dict (all weights)       │
    │    - config (architecture settings)        │
    │    - feature_names (72 features)          │
    │    - class_names ['Normal', 'Attack']     │
    └────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PHASE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    Real-Time Network Traffic
         │
         ├─ Flow 1: Normal web browsing
         ├─ Flow 2: DDoS attack (SYN flood)
         ├─ Flow 3: Normal SSH session
         └─ Flow 4: Port scan
         │
         ↓
    ┌────────────────────────────────────────────┐
    │       FEATURE EXTRACTION                   │
    │                                            │
    │  Monitoring Tool (e.g., Zeek, Suricata)   │
    │  Extracts: Flow Duration, Packet Counts,  │
    │            IAT Stats, Flags, etc.         │
    │                                            │
    │  ⚠️ PROBLEM: Some features missing!       │
    │     - Zeek: 50/72 features                │
    │     - Suricata: 35/72 features            │
    │     - Lightweight IDS: 15/72 features     │
    └────────────────┬───────────────────────────┘
                     │
                     ↓
    ┌────────────────────────────────────────────┐
    │       MoEInference.preprocess_features()   │
    │                                            │
    │  1. Handle Missing Features:              │
    │     - Strategy 'zero': Fill with 0        │
    │     - Strategy 'mean': Fill with mean     │
    │     - Strategy 'median': Fill with median │
    │                                            │
    │  2. Normalize:                             │
    │     - Use same scaler as training         │
    │                                            │
    │  3. Convert to tensor:                     │
    │     - Shape: [batch_size, 72]             │
    └────────────────┬───────────────────────────┘
                     │
                     ↓
    ┌────────────────────────────────────────────────────────────┐
    │                    MOE FORWARD PASS                         │
    │                                                             │
    │  Input: Normalized features [batch, 72]                    │
    │    │                                                        │
    │    ├──────────────────────┬─────────────────────────┐      │
    │    │                      │                         │      │
    │    ↓                      ↓                         ↓      │
    │  Split Features     Split Features           Concatenate   │
    │  (47 tabular)       (25 temporal)            (all 72)      │
    │    │                      │                         │      │
    │    ↓                      ↓                         │      │
    │ ┌──────────────┐   ┌──────────────┐               │      │
    │ │ FT-Transformer│   │   1D CNN    │                │      │
    │ │  (648k params)│   │ (27k params) │                │      │
    │ └──────┬───────┘   └──────┬───────┘               │      │
    │        │                   │                        │      │
    │        ↓                   ↓                        ↓      │
    │   [128-dim]            [128-dim]            ┌──────────────┐│
    │        │                   │                │ Gating Network││
    │        │                   │                │  (42k params) ││
    │        │                   │                └──────┬────────┘│
    │        │                   │                       │         │
    │        └──────────┬────────┘                       ↓         │
    │                   │                       [w_tab, w_temp]    │
    │                   ↓                         (softmax)        │
    │            Weighted Combine                                  │
    │       output = w_tab * tab_emb + w_temp * temp_emb          │
    │                   │                                          │
    │                   ↓                                          │
    │              [128-dim]                                       │
    │                   │                                          │
    │                   ↓                                          │
    │            ┌──────────────┐                                  │
    │            │  Classifier  │                                  │
    │            │ (MLP: 128→2) │                                  │
    │            └──────┬───────┘                                  │
    │                   │                                          │
    │                   ↓                                          │
    │              [logits: 2]                                     │
    │                   │                                          │
    │                   ↓                                          │
    │              Softmax                                         │
    │                   │                                          │
    │                   ↓                                          │
    │       [prob_normal, prob_attack]                             │
    └───────────────────┬─────────────────────────────────────────┘
                        │
                        ↓
    ┌────────────────────────────────────────────┐
    │       MoEInference.predict() OUTPUT        │
    │                                            │
    │  {                                         │
    │    'prediction': 'Attack',                 │
    │    'confidence': 0.987,                    │
    │    'probabilities': {                      │
    │      'Normal': 0.013,                      │
    │      'Attack': 0.987                       │
    │    },                                      │
    │    'gating_weights': {                     │
    │      'Tabular Expert': 0.22,               │
    │      'Temporal Expert': 0.78               │
    │    }                                       │
    │  }                                         │
    └────────────────┬───────────────────────────┘
                     │
                     ↓
    ┌────────────────────────────────────────────┐
    │         INTERPRETATION                     │
    │                                            │
    │  Gating = 22% Tabular, 78% Temporal       │
    │                                            │
    │  → Model is "Temporal-Dominant"           │
    │  → Decision driven by timing patterns     │
    │  → Likely attack: DDoS, Brute Force       │
    │                                            │
    │  Why?                                      │
    │  - Flow IAT Mean = 4.8ms (very low!)      │
    │  - SYN Flag Count = 250 (all SYN!)        │
    │  - Total Backward Packets = 0 (no reply)  │
    │                                            │
    │  → Classic SYN flood signature            │
    └────────────────────────────────────────────┘
```

---

## Missing Feature Handling Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SCENARIO: LIMITED MONITORING                         │
└─────────────────────────────────────────────────────────────────────────┘

Input: Only 10 features available (86% missing!)

  Feature Set (10/72 present):
  ┌──────────────────────────────────┐
  │ ✓ Flow Duration: 1200             │
  │ ✓ Total Fwd Packets: 250          │
  │ ✓ Flow IAT Mean: 4.8              │
  │ ✓ SYN Flag Count: 250             │
  │ ✓ ACK Flag Count: 0               │
  │ ✓ Destination Port: 80            │
  │ ✓ Total Backward Packets: 0       │
  │ ✓ Flow Packets/s: 208.3           │
  │ ✓ Flow IAT Std: 2.1               │
  │ ✓ Fwd Packets/s: 208.3            │
  │                                   │
  │ ✗ Total Length of Fwd Packets     │
  │ ✗ Total Length of Bwd Packets     │
  │ ✗ Fwd Packet Length Mean          │
  │ ✗ ... (58 more features missing)  │
  └──────────────────────────────────┘
                │
                ↓
  ┌──────────────────────────────────────────┐
  │  Missing Feature Handler                 │
  │                                          │
  │  Strategy: 'zero' (default)              │
  │                                          │
  │  For each missing feature:               │
  │    feature_value = 0.0                   │
  │                                          │
  │  Alternative strategies:                 │
  │    'mean': use training mean             │
  │    'median': use training median         │
  └──────────────────┬───────────────────────┘
                     │
                     ↓
  ┌──────────────────────────────────────────┐
  │  Complete Feature Vector (72 features)   │
  │                                          │
  │  [1200, 250, 4.8, 250, 0, 80, 0, 208.3,  │
  │   2.1, 208.3, 0, 0, 0, 0, ..., 0]        │
  │                                          │
  │  (10 real values + 62 zeros)             │
  └──────────────────┬───────────────────────┘
                     │
                     ↓
  ┌──────────────────────────────────────────┐
  │         FEATURE SPLITTING                │
  │                                          │
  │  Tabular (47):  [1200, 250, 80, ...]     │
  │    → 5 real + 42 zeros                   │
  │                                          │
  │  Temporal (25): [4.8, 2.1, 208.3, ...]   │
  │    → 5 real + 20 zeros                   │
  └──────────────────┬───────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
         ↓                        ↓
  ┌────────────┐          ┌────────────┐
  │ Tabular    │          │ Temporal   │
  │ Expert     │          │ Expert     │
  └─────┬──────┘          └─────┬──────┘
        │                       │
        ↓                       ↓
    [128-dim]               [128-dim]
    (weak signal)           (strong signal!)
        │                       │
        └──────────┬────────────┘
                   ↓
           ┌──────────────┐
           │   Gating     │
           │   Network    │
           └──────┬───────┘
                  │
                  ↓
        [Tabular: 0.22, Temporal: 0.78]
                  │
                  ↓
        🎯 MODEL ADAPTS AUTOMATICALLY!
           - Temporal features have signal
           - Tabular features mostly zeros
           - Gating shifts weight to Temporal Expert

  Result:
  ┌──────────────────────────────────────────┐
  │  Prediction: Attack                      │
  │  Confidence: 96.5%                       │
  │  Gating: 78% Temporal (adapted!)         │
  └──────────────────────────────────────────┘

  💡 Key Insight:
     Even with 86% missing features, the model still:
     ✓ Detects the attack (SYN flood)
     ✓ High confidence (96.5%)
     ✓ Adapts gating weights to available features
```

---

## Gating Network Decision Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOW GATING NETWORK DECIDES                           │
└─────────────────────────────────────────────────────────────────────────┘

Scenario 1: Normal Web Traffic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Flow Features:
    Flow Duration: 5234ms (normal)
    Total Fwd Packets: 3 (small)
    Flow IAT Mean: 1046ms (typical HTTP)
    SYN/ACK Flags: 1/4 (proper handshake)

  Expert Outputs:
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Tabular Expert      │  │  Temporal Expert     │
    │                      │  │                      │
    │  "Packet counts OK"  │  │  "Timing normal"     │
    │  "Flags normal"      │  │  "IAT variance low"  │
    │  "Port = 80 (HTTP)"  │  │  "No bursts"         │
    │                      │  │                      │
    │  Embedding: [...]    │  │  Embedding: [...]    │
    └──────────────────────┘  └──────────────────────┘

  Gating Input: Concatenate both embeddings [256-dim]

  Gating Network Decision:
    Input → FC(256→128) → ReLU → FC(128→64) → ReLU → FC(64→2) → Softmax
    
    Output: [0.52, 0.48]
    
    → Balanced! Both experts contribute equally
    → Both tabular and temporal patterns are normal

  Final Prediction: Normal (Confidence: 95%)


Scenario 2: DDoS SYN Flood Attack
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Flow Features:
    Flow Duration: 1200ms (very short)
    Total Fwd Packets: 250 (huge!)
    Flow IAT Mean: 4.8ms (extremely low - automated tool!)
    SYN Flag Count: 250 (all SYN, no ACK - red flag!)
    Total Backward Packets: 0 (server not responding)

  Expert Outputs:
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Tabular Expert      │  │  Temporal Expert     │
    │                      │  │                      │
    │  "High SYN count"    │  │  "IAT = 4.8ms!!!"    │
    │  "No backward pkts"  │  │  "Perfect timing"    │
    │  "Port = 80"         │  │  "Bot-like behavior" │
    │                      │  │                      │
    │  Embedding: [...]    │  │  Embedding: [...]    │
    │  (moderate signal)   │  │  (STRONG signal!)    │
    └──────────────────────┘  └──────────────────────┘

  Gating Network Decision:
    Input → FC(256→128) → ReLU → FC(128→64) → ReLU → FC(64→2) → Softmax
    
    Output: [0.22, 0.78]
    
    → Temporal-Dominant!
    → The 4.8ms IAT is the smoking gun
    → Automated attack tool signature

  Final Prediction: Attack (Confidence: 98.7%)


Scenario 3: Port Scan
━━━━━━━━━━━━━━━━━━━

  Flow Features:
    Flow Duration: 100ms (very short)
    Total Fwd Packets: 1 (single SYN)
    Flow IAT Mean: 0 (single packet)
    SYN Flag Count: 1
    Destination Port: 22 (SSH)

  Expert Outputs:
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Tabular Expert      │  │  Temporal Expert     │
    │                      │  │                      │
    │  "Unusual port combo"│  │  "Single packet"     │
    │  "No connection est."│  │  "No timing info"    │
    │  "SYN without ACK"   │  │  "IAT = 0"           │
    │                      │  │                      │
    │  Embedding: [...]    │  │  Embedding: [...]    │
    │  (STRONG signal!)    │  │  (weak signal)       │
    └──────────────────────┘  └──────────────────────┘

  Gating Network Decision:
    Output: [0.65, 0.35]
    
    → Tabular-Dominant!
    → Port/flag patterns are more informative
    → Not enough timing data

  Final Prediction: Attack (Confidence: 92%)
```

---

## Performance Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL PERFORMANCE COMPARISON                         │
└─────────────────────────────────────────────────────────────────────────┘

CICIDS Dataset Results:

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  XGBoost:           ████████████████████████████  99.8%         │
  │                                                                  │
  │  MoE (Ours):        ████████████████████████░░    98.3%         │
  │                                                                  │
  │  FT-Transformer:    ██████████████████████░░░░    96.4%         │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
               0%                50%               100%
                          F1 Score

  Trade-offs:

    XGBoost:
      ✓ Highest accuracy (99.8%)
      ✓ Fast training
      ✗ Low interpretability (just feature importance)
      ✗ Doesn't model expert specialization

    MoE (Ours):
      ✓ High interpretability (gating weights per sample!)
      ✓ Models expert specialization (tabular vs temporal)
      ✓ Robust to missing features
      ✗ Slightly lower accuracy (-1.5% vs XGBoost)

    FT-Transformer:
      ✓ Medium interpretability (attention weights)
      ✗ Lower accuracy (-1.9% vs MoE)
      ✗ Not specialized for temporal patterns


Missing Feature Robustness:

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  100% features:     ████████████████████████████  98.3%         │
  │                                                                  │
  │  90% features:      ███████████████████████████░  97.7%         │
  │                                                                  │
  │  80% features:      ██████████████████████████░░  96.8%         │
  │                                                                  │
  │  50% features:      ████████████████████░░░░░░░░  91.2%         │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
               0%                50%               100%
                          F1 Score

  Key Insight: With only 50% of features, still achieves 91.2% F1!


Inference Throughput:

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  CPU (single):      ███░░░░░░░░░░░░░░░░░░░░░░░  434 samples/s   │
  │                                                                  │
  │  CPU (batch=32):    ███████░░░░░░░░░░░░░░░░░░  1,024 samples/s  │
  │                                                                  │
  │  GPU (batch=256):   ███████████████████░░░░░░  2,560 samples/s  │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
               0            1,000          2,000         3,000
                       Samples per Second

  Real-World Implication:
    - Small network (1,000 flows/sec): CPU is enough
    - Medium network (2,000 flows/sec): GPU batch processing
    - Large network (10,000+ flows/sec): Distributed GPU cluster
```

---

## File Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROJECT FILE STRUCTURE                           │
└─────────────────────────────────────────────────────────────────────────┘

CORE IMPLEMENTATION FILES:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  src/models/temporal_expert.py (370 lines)                         │
│    └─ TemporalExpert (1D CNN architecture)                         │
│                                                                     │
│  src/models/gating_network.py (380 lines)                          │
│    └─ GatingNetwork (MLP + softmax)                                │
│                                                                     │
│  src/models/moe_model.py (540 lines)                               │
│    ├─ MixtureOfExperts (main wrapper)                              │
│    └─ load_moe_model() (helper function)                           │
│         │                                                           │
│         └─ DEPENDS ON:                                              │
│              - temporal_expert.py                                   │
│              - gating_network.py                                    │
│              - ft_transformer.py (Tabular Expert)                   │
│                                                                     │
│  src/data/feature_config.py (280 lines)                            │
│    ├─ CICIDS_TEMPORAL_INDICES                                      │
│    ├─ UNSW_TEMPORAL_INDICES                                        │
│    └─ get_feature_split()                                          │
│                                                                     │
│  src/data/preprocess.py (modified)                                 │
│    └─ Added feature splitting logic                                │
│         │                                                           │
│         └─ DEPENDS ON:                                              │
│              - feature_config.py                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

TRAINING FILES:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  src/models/train_moe.py (580 lines)                               │
│    ├─ pretrain_tabular_expert() (Stage 1)                          │
│    ├─ train_moe() (Stage 2)                                        │
│    └─ load_moe_dataset()                                           │
│         │                                                           │
│         └─ DEPENDS ON:                                              │
│              - moe_model.py                                         │
│              - feature_config.py                                    │
│              - preprocess.py                                        │
│                                                                     │
│  Usage:                                                             │
│    python -m src.models.train_moe --stage pretrain                 │
│    python -m src.models.train_moe --stage train                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

INFERENCE FILES:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  src/serving/inference.py (800 lines) ← NEW!                       │
│    ├─ MoEInference (main inference class)                          │
│    ├─ predict() (single sample)                                    │
│    ├─ predict_batch() (batch processing)                           │
│    ├─ preprocess_features() (missing feature handling)             │
│    └─ analyze_prediction() (interpretation)                        │
│         │                                                           │
│         └─ DEPENDS ON:                                              │
│              - moe_model.py                                         │
│              - feature_config.py                                    │
│              - models/weights/cicids_moe_best.pt (checkpoint)       │
│                                                                     │
│  Usage:                                                             │
│    from src.serving.inference import MoEInference                  │
│    model = MoEInference('CICIDS', 'models/weights/cicids_moe.pt')  │
│    result = model.predict(features)                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

ANALYSIS FILES:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  src/models/analyze_gating.py (420 lines)                          │
│    ├─ extract_gating_weights()                                     │
│    ├─ analyze_by_attack_type()                                     │
│    └─ visualize_gating_by_attack()                                 │
│         │                                                           │
│         └─ DEPENDS ON:                                              │
│              - moe_model.py                                         │
│              - preprocess.py                                        │
│                                                                     │
│  Usage:                                                             │
│    python -m src.models.analyze_gating                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

TEST FILES:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  test_inference.py (comprehensive demo) ← NEW!                     │
│    ├─ test_single_prediction()                                     │
│    ├─ test_missing_features()                                      │
│    ├─ test_batch_prediction()                                      │
│    └─ test_different_missing_strategies()                          │
│         │                                                           │
│         └─ DEPENDS ON:                                              │
│              - src/serving/inference.py                             │
│                                                                     │
│  Usage:                                                             │
│    python test_inference.py                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

DOCUMENTATION FILES:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  MOE_RESULTS.md (500+ lines)                                       │
│    - Training results, confusion matrices, architecture overview   │
│                                                                     │
│  MoE_CROSS_DATASET_ANALYSIS.md (400+ lines)                        │
│    - CICIDS vs UNSW comparison, gating behavior analysis           │
│                                                                     │
│  MOE_ARCHITECTURE_CODE_GUIDE.md (extensive)                        │
│    - Code snippets, forward pass walkthrough, technical details    │
│                                                                     │
│  INFERENCE_AND_MISSING_FEATURES_GUIDE.md (comprehensive) ← NEW!    │
│    - Production deployment, missing feature experiments            │
│                                                                     │
│  QUICK_START_INFERENCE.md (quick reference) ← NEW!                 │
│    - Quick start examples, FAQ, deployment patterns                │
│                                                                     │
│  COMPLETE_PROJECT_SUMMARY.md (this summary) ← NEW!                 │
│    - Complete project overview, all components listed              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

DATA FLOW:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Raw Data (data/raw/*.csv)                                         │
│        ↓                                                            │
│  preprocess.py → Processed Data (data/processed/)                  │
│        ↓                                                            │
│  train_moe.py → Model Checkpoint (models/weights/*.pt)             │
│        ↓                                                            │
│  inference.py → Predictions (Real-time)                            │
│        ↓                                                            │
│  Production API/Streaming → Alerts                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

**🎉 You now have a complete visual understanding of the entire MoE system!**

**Next:** Run `python test_inference.py` to see everything in action! 🚀
