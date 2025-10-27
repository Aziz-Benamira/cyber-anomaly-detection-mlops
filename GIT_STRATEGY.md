# Git Strategy for MLOps Project

## ✅ What IS Committed to Git

### Code & Configuration
- `src/` - All Python source code
- `scripts/` - Helper scripts
- `tests/` - Unit tests
- `notebooks/` - Jupyter notebooks (exploratory analysis)
- `params.yaml` - Model/training parameters
- `dvc.yaml` - DVC pipeline definition
- `environment.yml` - Conda environment
- `requirements.txt` - Python dependencies
- `README.md` and documentation

### Docker & Monitoring
- `docker/Dockerfile.*` - Docker images
- `docker/docker-compose.yml` - Docker orchestration
- `monitoring/*.yml` - Prometheus/Grafana configs

### DVC Metadata (NOT the data itself)
- `*.dvc` files - Pointers to data in remote storage
- `.dvc/config` - DVC configuration
- `dvc.lock` - Pipeline state

## ❌ What is NOT Committed (in .gitignore)

### Large Data Files
- ❌ `data/raw/*.csv` - Raw datasets (GB-sized)
- ❌ `data/processed/**/*.npy` - Processed numpy arrays
- ❌ `data/processed/**/*.joblib` - Sklearn preprocessors
- ✅ **Use DVC instead** for versioning these files

### Model Artifacts
- ❌ `models/weights/*.pt` - PyTorch model weights
- ❌ `mlruns/` - MLflow tracking data
- ✅ **Use MLflow Model Registry** or DVC for model versioning

### Environment & Secrets
- ❌ `.env` files - API keys, credentials
- ❌ `venv/`, `__pycache__/` - Python artifacts
- ❌ `.vscode/`, `.idea/` - IDE settings

### Temporary & Generated
- ❌ `*.log` - Log files
- ❌ `.ipynb_checkpoints/` - Notebook checkpoints
- ❌ `reports/*.json` - Generated reports

## 🔄 DVC vs Git Strategy

| Type | Tool | Why |
|------|------|-----|
| **Code** | Git | Small, text-based, frequent changes |
| **Data** | DVC | Large binary files, versioned separately |
| **Models** | DVC + MLflow | Track experiments + version artifacts |
| **Configs** | Git | Parameters should be version controlled |

## 📝 Recommended Workflow

### First Time Setup
```bash
# Initialize Git (if not done)
git init

# Add files
git add .gitignore README.md src/ scripts/ params.yaml dvc.yaml environment.yml

# Commit
git commit -m "Initial project setup"

# Add remote
git remote add origin https://github.com/Aziz-Benamira/cyber-anomaly-detection-mlops.git

# Push
git push -u origin main
```

### After Creating Mini Datasets
```bash
# Add data to DVC (not Git!)
dvc add data/processed/cicids
dvc add data/processed/unsw

# This creates .dvc files - commit THESE to Git
git add data/processed/cicids.dvc data/processed/unsw.dvc .gitignore
git commit -m "Track processed datasets with DVC"

# Push data to DVC remote (configure first)
dvc remote add -d storage s3://mybucket/dvcstore
dvc push

# Push metadata to Git
git push
```

### After Training Models
```bash
# Track model with DVC
dvc add models/weights/cicids_finetuned.pt

# Or log to MLflow Model Registry
mlflow models register ...

# Commit DVC metadata
git add models/weights/cicids_finetuned.pt.dvc
git commit -m "Add finetuned CICIDS model"
git push

# Push model to DVC remote
dvc push
```

## 🚨 Before Pushing - Checklist

Run these commands to verify:

```bash
# 1. Check file sizes (should all be < 100MB)
git ls-files | xargs ls -lh | awk '$5 ~ /M|G/ {print $5, $9}' | sort -rh

# 2. Verify CSV files are ignored
git check-ignore data/raw/*.csv

# 3. Check what will be pushed
git status
git diff --cached --stat

# 4. Verify no secrets
git diff --cached | grep -i "password\|api_key\|secret\|token"
```

## 💡 Pro Tips

1. **Never commit large files** - Use DVC or Git LFS
2. **Keep secrets out** - Use environment variables
3. **DVC for data** - Version data separately from code
4. **MLflow for experiments** - Track runs, don't commit mlruns/
5. **Meaningful commits** - Small, focused commits with clear messages

## 📦 File Size Limits

- **Git**: < 100 MB per file (GitHub limit)
- **Large files**: Use DVC or Git LFS
- **Typical sizes in this project**:
  - Raw CSV: 500MB - 2GB ❌ Don't commit!
  - Processed .npy: 50-500MB ❌ Don't commit!
  - Model .pt: 1-10MB ⚠️ Use DVC
  - Source code: < 1MB ✅ Commit

## 🔗 Resources

- [DVC Documentation](https://dvc.org/doc)
- [Git Best Practices](https://git-scm.com/book/en/v2)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
