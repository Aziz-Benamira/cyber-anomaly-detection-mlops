# Mini Datasets for Fast Experimentation

## 🎯 Purpose
For **portfolio/demo projects** and **local development**, the full CICIDS (3M rows) and UNSW datasets are too large. This setup creates **balanced mini datasets** for:
- ✅ Fast training iterations (~15 sec/epoch vs 16 min/epoch)
- ✅ Showcasing MLOps best practices without heavy compute
- ✅ Quick prototyping and debugging
- ✅ Local GPU-friendly experimentation

---

## 🐛 Bug Fixed: 100% Accuracy Issue

### **Problem Identified:**
The original training showed suspicious **100% accuracy from epoch 1**, caused by:

1. **❌ Labels were strings** - `y.npy` contained `['BENIGN', 'DDoS', ...]` instead of integers
2. **❌ Severe class imbalance** - 73% BENIGN, 27% attacks
3. **❌ No stratified splitting** - Random split could put all of one class in train/val

### **Solution Implemented:**
1. **✅ Fixed preprocessing** - Labels now properly encoded as `int64` (0=Normal, 1=Attack)
2. **✅ Balanced sampling** - 50/50 split between classes
3. **✅ Stratified train/val split** - Ensures both classes represented in train/val sets

### **Results After Fix:**
```
Before: Train=100% | Val=100% (suspicious - model not learning)
After:  Train=94.6% | Val=95.5% (realistic - proper learning curve!)
```

---

## 📦 Creating Mini Datasets

### **Method 1: Direct Command**
```bash
# CICIDS mini dataset (50k samples, balanced)
python -m src.data.preprocess --params params.yaml --mini --max-samples 50000

# UNSW mini dataset (50k samples, balanced)
# First update params.yaml to set dataset: "UNSW"
python -m src.data.preprocess --params params.yaml --mini --max-samples 50000
```

### **Method 2: Helper Script** (Recommended)
```bash
# Create CICIDS mini dataset
python scripts/create_mini_datasets.py --dataset CICIDS --samples 50000

# Create UNSW mini dataset
python scripts/create_mini_datasets.py --dataset UNSW --samples 50000

# Create both at once
python scripts/create_mini_datasets.py --all --samples 50000
```

---

## 📊 Dataset Comparison

| Metric | Original CICIDS | Mini CICIDS | Speedup |
|--------|----------------|-------------|---------|
| **Samples** | 3,088,416 | 50,000 | 62x smaller |
| **Class Balance** | 73% / 27% | 50% / 50% | ✅ Balanced |
| **Pretrain Time** | ~2 hours | ~80 seconds | **90x faster** |
| **Finetune/Epoch** | ~16 minutes | ~15 seconds | **64x faster** |
| **Val Accuracy** | N/A | 95.5% | Realistic |

---

## 🏃 Training Workflow

### **Step 1: Create Mini Dataset**
```bash
python scripts/create_mini_datasets.py --dataset CICIDS --samples 50000
```

**Output:**
```
[INFO] Label encoding applied:
       Original labels: ['BENIGN', 'DDoS', 'PortScan', 'Bot', ...]
       Encoded distribution: [2273097  815319]
       Class 0 (Normal): 2273097, Class 1 (Attack): 815319
[INFO] Balanced sampling: 50000 samples
       Class 0: 25000, Class 1: 25000
[INFO] Saved processed data to data\processed\cicids
       X shape = (50000, 72), y shape = (50000,)
```

### **Step 2: Pretrain (Self-Supervised MFM)**
```bash
python -m src.models.train_tabular --params params.yaml --stage pretrain
```

**Expected Time:** ~80 seconds  
**Output:** `models/weights/cicids_pretrained.pt`

### **Step 3: Finetune (Supervised Classification)**
```bash
python -m src.models.train_tabular --params params.yaml --stage finetune
```

**Expected Time:** ~5 epochs x 15 sec = 75 seconds  
**Output:** `models/weights/cicids_finetuned.pt`

**Training Progress:**
```
[Epoch 1] Train Acc=0.775 | Val Acc=0.911
[Epoch 2] Train Acc=0.921 | Val Acc=0.942
[Epoch 3] Train Acc=0.934 | Val Acc=0.947
[Epoch 4] Train Acc=0.941 | Val Acc=0.950
[Epoch 5] Train Acc=0.946 | Val Acc=0.955  ← Final
```

---

## 🔧 Code Changes Made

### **1. Fixed `src/data/preprocess.py`**
- Added proper label encoding (string → int64)
- Added balanced sampling function
- Added `--mini` flag for easy mini dataset creation
- Added class distribution logging

### **2. Fixed `src/models/train_tabular.py`**
- Replaced `random_split` with **stratified split** using `train_test_split`
- Added validation checks for label types
- Added class distribution logging
- Better error messages for debugging

### **3. Created `scripts/create_mini_datasets.py`**
- Convenience script for batch mini dataset creation
- Handles both CICIDS and UNSW datasets
- Automatic params.yaml updating

---

## 💡 Best Practices for Recruiters/Portfolio

### **Why Mini Datasets Are Perfect for Demos:**
1. **Fast iterations** - Test MLOps pipeline changes in minutes, not hours
2. **Reproducible** - Anyone can run your project locally
3. **Clear metrics** - Realistic accuracy shows proper model learning
4. **Cost-effective** - No need for expensive GPU hours
5. **Complete pipeline** - Still demonstrates end-to-end MLOps (DVC, MLflow, Docker)

### **What This Demonstrates:**
- ✅ Data preprocessing and feature engineering
- ✅ Class imbalance handling
- ✅ Two-stage training (pretrain + finetune)
- ✅ Experiment tracking (MLflow)
- ✅ Model versioning and checkpointing
- ✅ Debugging and problem-solving skills
- ✅ Performance optimization

---

## 📈 Next Steps

Now that you have fast, balanced datasets:

1. **Add DVC stages** for reproducible pipeline
2. **Implement evaluation metrics** (confusion matrix, precision, recall, F1)
3. **Build FastAPI endpoint** for model serving
4. **Add unit tests** for data and model
5. **Dockerize** training and serving
6. **Set up monitoring** with Prometheus/Grafana

All these can now be tested quickly with mini datasets! 🚀

---

## 🎓 For Recruiters

This project showcases:
- **Problem-solving**: Identified and fixed 100% accuracy bug
- **Data engineering**: Balanced sampling, stratified splitting, label encoding
- **MLOps**: DVC, MLflow, reproducible pipelines
- **Model development**: Transformer-based TabTransformer architecture
- **Performance**: 60x speedup while maintaining model quality
- **Documentation**: Clear README and code comments

**Training time:** Full pipeline (pretrain + finetune) completes in **~3 minutes** on local GPU.
