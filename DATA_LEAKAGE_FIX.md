# Data Leakage Bug Report - UNSW Dataset

## Issue Summary

**Severity**: CRITICAL  
**Date Discovered**: October 28, 2025  
**Status**: FIXED ✅

## The Problem

The UNSW-NB15 dataset preprocessing included the `attack_cat` feature as a categorical variable for model training. This caused **severe data leakage** because `attack_cat` directly reveals the label:

- `attack_cat = 'Normal'` → `label = 0` (normal traffic)
- `attack_cat = 'DoS'`, `'Exploits'`, `'Reconnaissance'`, etc. → `label = 1` (attack)

## Impact

The model achieved **100% validation accuracy from epoch 1**, which appeared impressive but was actually **invalid**:

```
Epoch 1: Val Acc = 100%
Epoch 2: Val Acc = 100%
...
Epoch 10: Val Acc = 100%
```

**This was not machine learning - the model simply memorized:**
```python
if attack_cat != 'Normal':
    predict 1 (attack)
else:
    predict 0 (normal)
```

## Root Cause Analysis

### File: `src/data/schema_unsw.py`

**Before (BROKEN)**:
```python
cat_cols: List[str] = (
    "proto", "service", "state", "attack_cat"  # ❌ attack_cat is leakage!
)
```

**After (FIXED)**:
```python
cat_cols: List[str] = (
    "proto", "service", "state"  # ✅ Removed attack_cat
)
```

## Evidence

Checking raw UNSW data confirms the leakage:

```python
>>> df[df['label']==0]['attack_cat'].unique()
array(['Normal'])

>>> df[df['label']==1]['attack_cat'].unique()
array(['Reconnaissance', 'Backdoor', 'DoS', 'Exploits', 'Analysis', 
       'Fuzzers', 'Shellcode', 'Worms', 'Generic'])
```

**Perfect 1:1 mapping** between `attack_cat` and `label`!

## Fix Applied

1. **Removed leaky feature** from `schema_unsw.py`
2. **Reprocessed dataset**: 202 features → 191 features
3. **Retrained model** with legitimate features only

## Results Comparison

| Metric | With Leakage (INVALID) | Without Leakage (VALID) |
|--------|------------------------|-------------------------|
| Features | 42 (39 num + 3 cat + **attack_cat**) | 42 (39 num + 3 cat) |
| Encoded Features | 202 | 191 |
| Epoch 1 Val Acc | 100% | 89.1% |
| Epoch 2 Val Acc | 100% | 90.4% |
| Final Val Acc | **100%** ❌ | **90.7%** ✅ |
| Model Behavior | Memorization | Learning |

## Validation

The fixed model shows **realistic learning behavior**:
- Starts at 83.3% train / 89.1% val (epoch 1)
- Gradually improves through training
- Converges to 91.9% train / 90.7% val (epoch 10)
- Small train-val gap indicates good generalization

## Lessons Learned

### 1. Always inspect categorical features
Features like `attack_cat`, `attack_type`, `label_category` are **red flags** for data leakage.

### 2. 100% accuracy is suspicious
When a model achieves perfect accuracy immediately:
- ✅ Check for data leakage
- ✅ Verify feature engineering
- ✅ Inspect categorical feature values
- ❌ Don't celebrate yet!

### 3. Domain knowledge matters
In cybersecurity datasets, attack category/type columns are **metadata for analysis**, not **legitimate features for prediction**.

### 4. Validate schema design
Before training:
- Review all features used
- Check correlation between features and labels
- Ask: "Would this feature be available at prediction time?"

## Prevention Checklist

For future datasets, verify:

- [ ] No features that directly encode the label
- [ ] No attack type/category columns as features
- [ ] No metadata columns mixed with predictive features
- [ ] Features represent information available **before** classification
- [ ] Suspicious 100% accuracy triggers investigation

## Files Modified

1. ✅ `src/data/schema_unsw.py` - Removed `attack_cat` from `cat_cols`
2. ✅ `data/processed/unsw/` - Reprocessed with 191 features (was 202)
3. ✅ `models/weights/unsw_*.pt` - Retrained models without leakage
4. ✅ `FT_TRANSFORMER_RESULTS.md` - Updated with correct metrics

## Final Status

- **UNSW Dataset**: Fixed and validated ✅
- **CICIDS Dataset**: No leakage detected ✅
- **Training Results**: Realistic and trustworthy ✅

The project now has **clean, valid models** ready for evaluation and deployment.
