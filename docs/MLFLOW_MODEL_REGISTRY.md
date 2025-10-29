# 📚 MLflow Model Registry - Complete Guide

## 🎯 What is MLflow Model Registry?

**MLflow Model Registry** is a centralized model store that manages the **full lifecycle** of ML models:

- 📦 **Version Control** for models (like Git for code)
- 🏷️ **Staging System** (Development → Staging → Production)
- 📝 **Metadata Tracking** (who trained it, when, performance metrics)
- 🔄 **Model Lineage** (which data/code version produced this model)
- 🚀 **Deployment Management** (promote/rollback models)

### 💡 Why Do We Need It?

**Without Model Registry:**
```
❌ Models saved as files: model_v1.pt, model_v2_final.pt, model_v2_FINAL_FINAL.pt
❌ No idea which model is in production
❌ Can't track which data trained which model
❌ Manual deployment process (error-prone)
❌ No rollback if model fails
```

**With Model Registry:**
```
✅ Models versioned automatically: v1, v2, v3...
✅ Clear stages: Staging → Production
✅ Full lineage: data version + code version + metrics
✅ One-click deployment
✅ Easy rollback to previous version
```

---

## 🏗️ MLflow Model Registry Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLFLOW TRACKING SERVER                       │
│                                                                 │
│  Experiments (train runs) ──┐                                  │
│                              │                                  │
│  Run ID: abc123             │                                  │
│    - Parameters             │                                  │
│    - Metrics (F1=98.3%)     │                                  │
│    - Artifacts (model.pt)   │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                               │ Register Model
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY                               │
│                                                                 │
│  Model Name: "moe-cybersecurity"                               │
│    │                                                            │
│    ├─ Version 1                                                │
│    │    Stage: Archived                                        │
│    │    F1 Score: 96.4%                                        │
│    │    Created: 2025-10-15                                    │
│    │                                                            │
│    ├─ Version 2                                                │
│    │    Stage: Staging                                         │
│    │    F1 Score: 98.3%                                        │
│    │    Created: 2025-10-20                                    │
│    │    Tags: {dataset: CICIDS, architecture: MoE}             │
│    │                                                            │
│    └─ Version 3                                                │
│         Stage: Production  ← Currently serving                 │
│         F1 Score: 98.5%                                        │
│         Created: 2025-10-28                                    │
│         Approved by: Data Science Team                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ Load for Serving
                               ↓
                        ┌──────────────┐
                        │  API Server  │
                        │              │
                        │ model = load │
                        │  (prod, v3)  │
                        └──────────────┘
```

---

## 📖 Key Concepts

### 1️⃣ Model Versions
Every time you register a model, MLflow creates a **new version**:
- Version 1, 2, 3, 4...
- Each version is **immutable** (can't change once created)
- Each version has its own metrics, parameters, artifacts

### 2️⃣ Model Stages
Each model version can be in one of these stages:

| Stage | Purpose | Example Use Case |
|-------|---------|------------------|
| **None** | Just registered | Initial model training |
| **Staging** | Testing phase | Model being validated by QA team |
| **Production** | Live deployment | Model serving real traffic |
| **Archived** | Deprecated | Old models no longer needed |

### 3️⃣ Model Metadata
Each model version stores:
- **Source Run:** Which experiment run created it
- **Parameters:** Hyperparameters used
- **Metrics:** Performance (F1, accuracy, etc.)
- **Tags:** Custom labels (dataset=CICIDS, architecture=MoE)
- **Description:** Human-readable notes

### 4️⃣ Model Lineage
Track the complete history:
```
Data Version (DVC) → Code Version (Git) → Training Run (MLflow) → Model Version (Registry)
    ↓                      ↓                      ↓                      ↓
  v1.0.0              commit abc123          Run: exp_001          Model: v3
```

---

## 🛠️ How to Use MLflow Model Registry

### Step 1: Register a Model

**Option A: During Training (Automatic)**
```python
import mlflow
import mlflow.pytorch

# Start MLflow run
with mlflow.start_run(run_name="moe_training"):
    # Train model
    model = train_moe_model()
    
    # Log parameters
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("batch_size", 256)
    
    # Log metrics
    mlflow.log_metric("f1_score", 0.983)
    mlflow.log_metric("precision", 0.971)
    
    # Register model (creates new version automatically)
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name="moe-cybersecurity"  # ← This registers it!
    )
```

**Option B: After Training (Manual)**
```python
import mlflow

# Load existing run
run_id = "abc123def456"  # From your training run

# Register the model from that run
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(
    model_uri=model_uri,
    name="moe-cybersecurity"
)
```

### Step 2: Manage Model Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model to Staging
client.transition_model_version_stage(
    name="moe-cybersecurity",
    version=2,
    stage="Staging"
)

# After validation, promote to Production
client.transition_model_version_stage(
    name="moe-cybersecurity",
    version=2,
    stage="Production"
)

# Archive old production model
client.transition_model_version_stage(
    name="moe-cybersecurity",
    version=1,
    stage="Archived"
)
```

### Step 3: Add Metadata

```python
# Add description
client.update_model_version(
    name="moe-cybersecurity",
    version=2,
    description="MoE model trained on CICIDS dataset. F1=98.3%. Uses 1D CNN temporal expert."
)

# Add tags
client.set_model_version_tag(
    name="moe-cybersecurity",
    version=2,
    key="dataset",
    value="CICIDS"
)

client.set_model_version_tag(
    name="moe-cybersecurity",
    version=2,
    key="architecture",
    value="MoE-FTTransformer-CNN"
)
```

### Step 4: Load Model for Inference

```python
import mlflow.pyfunc

# Option A: Load latest production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/moe-cybersecurity/Production"
)

# Option B: Load specific version
model = mlflow.pyfunc.load_model(
    model_uri="models:/moe-cybersecurity/2"
)

# Use for prediction
predictions = model.predict(data)
```

---

## 🔍 MLflow UI - Model Registry Tab

Access the MLflow UI:
```bash
mlflow ui --port 5000
```

Then navigate to: **http://localhost:5000**

**What you'll see:**

1. **Models Tab:**
   - List of all registered models
   - Click on "moe-cybersecurity" to see versions

2. **Model Details:**
   - All versions (1, 2, 3...)
   - Current stage for each
   - Metrics and parameters
   - Source run link

3. **Version Details:**
   - Full metadata
   - Download model artifacts
   - Transition stage buttons
   - Edit description/tags

---

## 🚀 Our Implementation

We'll register our MoE model with this workflow:

### Workflow Diagram

```
Training Script (train_moe.py)
        ↓
    Train Model
        ↓
    Log to MLflow
        ↓
    Register Model
        ↓
┌───────────────────────┐
│  MLflow Registry      │
│                       │
│  moe-cybersecurity-   │
│  cicids               │
│    v1: Staging        │
│    v2: Production ✓   │
└───────────────────────┘
        ↓
    Load in API
        ↓
    Serve Predictions
```

### Models We'll Register

1. **moe-cybersecurity-cicids**
   - MoE model trained on CICIDS
   - Versions: Pretrained tabular, Full MoE
   - Stage: Production

2. **moe-cybersecurity-unsw**
   - MoE model trained on UNSW
   - Versions: Pretrained tabular, Full MoE
   - Stage: Staging (for validation)

---

## 📝 Best Practices

### 1. **Naming Convention**
```python
# Good: Descriptive and consistent
"moe-cybersecurity-cicids"
"ft-transformer-unsw"
"xgboost-baseline-cicids"

# Bad: Unclear or inconsistent
"model1"
"my_model_final_v2"
"test_model"
```

### 2. **Stage Transitions**
```
None → Staging → Production
               ↓
           Archived (when replaced)
```

**Never skip staging!** Always validate before production.

### 3. **Metadata Management**
```python
# Always add:
- Description (what is this model?)
- Tags (dataset, architecture, date)
- Performance metrics (F1, precision, recall)
- Training parameters (learning rate, epochs)
```

### 4. **Version Control**
```python
# Keep track of:
- Git commit hash (code version)
- DVC data version (data version)
- MLflow run ID (experiment)
- Model version (registry)
```

---

## 🎓 Common Operations

### Check Current Production Model
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get production versions
prod_versions = client.get_latest_versions(
    name="moe-cybersecurity-cicids",
    stages=["Production"]
)

for version in prod_versions:
    print(f"Version: {version.version}")
    print(f"Run ID: {version.run_id}")
    print(f"Status: {version.status}")
```

### Compare Model Versions
```python
# Get all versions
versions = client.search_model_versions(
    filter_string="name='moe-cybersecurity-cicids'"
)

# Compare metrics
for v in versions:
    run = client.get_run(v.run_id)
    f1 = run.data.metrics.get("f1_score", 0)
    print(f"Version {v.version}: F1={f1:.4f}, Stage={v.current_stage}")
```

### Rollback to Previous Version
```python
# If new production model fails, rollback
client.transition_model_version_stage(
    name="moe-cybersecurity-cicids",
    version=2,  # Old version
    stage="Production"
)

client.transition_model_version_stage(
    name="moe-cybersecurity-cicids",
    version=3,  # New broken version
    stage="Archived"
)
```

---

## 🔧 Integration with Our Pipeline

### In Training Script
```python
# src/models/train_moe.py

import mlflow
import mlflow.pytorch

def train_moe(dataset_name='CICIDS'):
    # Set experiment
    mlflow.set_experiment(f"moe-training-{dataset_name.lower()}")
    
    with mlflow.start_run(run_name=f"moe_{dataset_name}"):
        # Log parameters
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("architecture", "MoE")
        mlflow.log_param("temporal_expert", "1D-CNN")
        
        # Train model
        model = train_model()
        
        # Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        
        # Register model
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"moe-cybersecurity-{dataset_name.lower()}"
        )
```

### In Inference API
```python
# src/serving/api.py

import mlflow.pyfunc

# Load production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/moe-cybersecurity-cicids/Production"
)

# Use for predictions
@app.post("/predict")
def predict(features: dict):
    result = model.predict(features)
    return result
```

---

## 🎯 Summary

**What is MLflow Model Registry?**
- Centralized storage for ML models
- Version control for models
- Stage management (Staging → Production)
- Metadata and lineage tracking

**Why use it?**
- ✅ No more "model_final_v2_FINAL.pt"
- ✅ Easy rollback if model fails
- ✅ Track which data trained which model
- ✅ Automate deployment workflows

**Key Operations:**
1. Register model → Creates new version
2. Transition stage → Move to Staging/Production
3. Add metadata → Tags, descriptions, metrics
4. Load model → Use in production API

**Next:** We'll implement model registration in our training pipeline!

---

## 📚 Additional Resources

- [MLflow Model Registry Docs](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/mlflow.html)
- [Best Practices Guide](https://mlflow.org/docs/latest/model-registry.html#model-registry-workflows)

**Ready to implement? Let's register our MoE models!** 🚀
