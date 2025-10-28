# FT-Transformer Architecture Migration

## Executive Summary

**Date**: January 2025  
**Status**: ✅ COMPLETE  
**Impact**: CRITICAL - Foundational architecture upgrade

Successfully migrated from **TabTransformer** to **FT-Transformer** architecture to fix a critical limitation: TabTransformer only applies self-attention to categorical features, while numerical features bypass the transformer entirely. Since CICIDS is 100% numerical and UNSW is predominantly numerical (39/43 features), this severely limited model expressiveness.

---

## The Problem

### TabTransformer Architecture (OLD)

```
Input: X_num (numerical), X_cat (categorical)
  ↓
Categorical features → Embeddings → Transformer → Attention-enriched representations
Numerical features   → [BYPASS] → Directly concatenated (no attention)
  ↓
Concatenate [attended_categorical, raw_numerical] → MLP → Output
```

**Critical Flaw**:
- CICIDS: 72 numerical features, 0 categorical → **Transformer gets ZERO features**
- UNSW: 39 numerical, 4 categorical → **Transformer only sees 4/43 features**
- Numerical features never interact via self-attention
- Model degenerates to a simple MLP for numerical-heavy datasets

---

## The Solution

### FT-Transformer Architecture (NEW)

```
Input: X_num (numerical), X_cat (categorical)
  ↓
Numerical features:   token_j = b_j + x_j * W_j  (per-feature linear projection)
Categorical features: token_j = Embedding(x_j)   (standard lookup)
  ↓
All tokens → Prepend [CLS] → Transformer → Self-attention across ALL features
  ↓
Extract [CLS] token → MLP → Output
```

**Key Improvements**:
- **ALL features tokenized**: Numericals get per-feature linear projections
- **Unified attention**: All features interact via self-attention
- **Rich representations**: Numerical features can attend to each other
- **Dataset agnostic**: Works equally well for numerical-only, categorical-only, or mixed datasets

---

## Implementation Details

### 1. New `FTFeatureTokenizer` Class

**Numerical Tokenization** (NEW):
```python
def tokenize_numerical(self, x_num: torch.Tensor) -> torch.Tensor:
    """
    Per-feature linear projection: token_j = b_j + x_j * W_j
    
    Args:
        x_num: (B, n_num) - normalized numerical features
    Returns:
        (B, n_num, d_model) - tokenized numerical features
    """
    x_expanded = x_num.unsqueeze(-1)  # (B, n_num, 1)
    tokens = x_expanded * self.num_weight.unsqueeze(0)  # Broadcasting
    if self.use_bias:
        tokens = tokens + self.num_bias.unsqueeze(0)
    return tokens
```

**Categorical Tokenization** (UNCHANGED):
```python
def tokenize_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
    """
    Standard embedding lookup for categorical features
    
    Args:
        x_cat: (B, n_cat) - categorical feature indices
    Returns:
        (B, n_cat, d_model) - tokenized categorical features
    """
    tokens = []
    for i, embed_layer in enumerate(self.cat_embeddings):
        tokens.append(embed_layer(x_cat[:, i]))  # (B, d_model)
    return torch.stack(tokens, dim=1)  # (B, n_cat, d_model)
```

**Combined Forward**:
```python
def forward(self, x_num=None, x_cat=None) -> torch.Tensor:
    """
    Tokenize all features and prepend [CLS] token
    
    Returns:
        (B, 1 + n_features, d_model)
        - Position 0: [CLS] token (learnable)
        - Positions 1 to n_num: Numerical feature tokens
        - Positions n_num+1 to n_features: Categorical feature tokens
    """
    all_tokens = torch.cat([num_tokens, cat_tokens], dim=1)
    cls = self.cls_token.expand(B, -1, -1)
    return torch.cat([cls, all_tokens], dim=1)
```

### 2. Updated `TabularExpert` Methods

**Classification** (now uses [CLS] token):
```python
def forward_classify(self, X_num=None, X_cat=None) -> torch.Tensor:
    tokens = self.tokenizer(x_num=X_num, x_cat=X_cat)  # (B, 1+n_features, d_model)
    h = self.backbone(tokens)                           # Transformer encoding
    cls_embedding = self._extract_cls_token(h)          # (B, d_model)
    logits = self.clf_head(cls_embedding)               # (B, n_classes)
    return logits
```

**Masked Feature Modeling** (updated tokenization):
```python
def forward_mfm(self, X_num=None, X_cat=None) -> Tuple[torch.Tensor, Dict]:
    tokens = self.tokenizer(x_num=X_num, x_cat=X_cat)  # All features tokenized
    mask_idx = self._sample_mask(B, n_features, device)
    tokens_masked = self.tokenizer.apply_mask(tokens, mask_idx)
    h = self.backbone(tokens_masked)
    loss, logs = self.mfm_head(h, mask_idx, X_num, X_cat)
    return loss, logs
```

---

## Validation Results

### Test Suite (all tests passed ✅)

**Test 1: CICIDS-like (72 numerical, 0 categorical)**
```
Classification shape: torch.Size([4, 2]) ✅
MFM loss: 1.1227 ✅
```

**Test 2: UNSW-like (39 numerical, 4 categorical)**
```
Classification shape: torch.Size([4, 2]) ✅
MFM loss: 1.1463 ✅
```

**Test 3: Tokenization mechanics**
```
Input:  (2, 72)       - 2 samples, 72 features
Output: (2, 73, 128)  - [CLS] + 72 feature tokens, 128-dim embeddings ✅
```

---

## Impact on Datasets

### CICIDS2017
- **Before**: 72 numerical features → bypassed transformer → MLP only
- **After**: 72 numerical features → tokenized → full self-attention
- **Improvement**: Transformer now learns from ALL features instead of being unused

### UNSW-NB15
- **Before**: 39 numerical (bypassed) + 4 categorical (attended)
- **After**: 39 numerical (attended) + 4 categorical (attended)
- **Improvement**: 10x more features participate in attention mechanism

---

## Files Modified

### `src/models/tabular_expert.py`
- **Added**: `FTFeatureTokenizer` class (120+ lines)
  - `tokenize_numerical()`: Per-feature linear projection
  - `tokenize_categorical()`: Standard embeddings
  - `forward()`: Concatenate + prepend [CLS]
  - `apply_mask()`: Masking for MFM pretraining
  
- **Updated**: `TabularExpertConfig`
  - Added `numerical_embedding_bias` parameter
  - Removed `init_feature_name_embed` (not needed in FT-Transformer)
  
- **Updated**: `TabularExpert` class
  - Constructor now uses `FTFeatureTokenizer`
  - `forward_classify()`: Uses `_extract_cls_token()` instead of `_pool()`
  - `forward_mfm()`: Updated to new tokenizer API
  
- **Removed**: Old `FeatureTokenizer` class (TabTransformer-style)
- **Removed**: `_pool()` method (replaced by `_extract_cls_token()`)

---

## Backward Compatibility

⚠️ **BREAKING CHANGE**: Existing pretrained models are incompatible

**Reason**: 
- Old models: Numerical features were NOT tokenized (different architecture)
- New models: Numerical features ARE tokenized (new parameters: `num_weight`, `num_bias`)

**Action Required**:
1. Delete old checkpoint files: `models/weights/*.pt`
2. Retrain from scratch using new FT-Transformer architecture
3. Update any saved model references

---

## Next Steps

### Immediate (Required)
1. ✅ Test architecture with synthetic data (COMPLETE)
2. ⏳ Retrain CICIDS model with FT-Transformer
3. ⏳ Retrain UNSW model with FT-Transformer
4. ⏳ Compare performance: TabTransformer vs FT-Transformer

### Short-term (Recommended)
1. Benchmark attention patterns: Verify numerical features attend to each other
2. Ablation study: Quantify impact of numerical tokenization
3. Update documentation with architecture diagrams
4. Add unit tests for `FTFeatureTokenizer`

### Long-term (Optional)
1. Experiment with different tokenization schemes (learned scaling, etc.)
2. Visualize attention maps to understand feature interactions
3. Compare with other tabular architectures (TabNet, SAINT, etc.)

---

## References

**Paper**: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
- Introduced FT-Transformer architecture
- Showed per-feature tokenization outperforms TabTransformer on numerical-heavy datasets
- Demonstrated importance of [CLS] token for aggregation

**Key Insight**: Tabular data often has rich numerical structure that requires attention-based interaction, not just MLP concatenation.

---

## Migration Checklist

- [x] Implement `FTFeatureTokenizer` class
- [x] Update `TabularExpert.__init__()` to use new tokenizer
- [x] Update `forward_classify()` to extract [CLS] token
- [x] Update `forward_mfm()` to use new tokenizer API
- [x] Remove old `FeatureTokenizer` class
- [x] Remove `_pool()` method
- [x] Fix duplicate code in mask sampling
- [x] Test with CICIDS-like data (numerical-only)
- [x] Test with UNSW-like data (mixed numerical+categorical)
- [x] Verify tokenization shapes
- [x] Validate no linting errors
- [ ] Retrain CICIDS model
- [ ] Retrain UNSW model
- [ ] Performance comparison
- [ ] Update PROJECT_STATUS.md

---

## Summary

This migration addresses a **critical architectural flaw** that prevented the transformer from learning from numerical features. The FT-Transformer now provides:

1. **Unified feature processing**: All features tokenized and attended
2. **Rich numerical interactions**: Self-attention across numerical features
3. **Dataset flexibility**: Works for numerical-only, categorical-only, or mixed data
4. **Theoretically sound**: Follows state-of-the-art tabular deep learning practices

The codebase is now ready for retraining with the improved architecture.
