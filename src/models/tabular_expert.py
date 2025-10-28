"""
FT-Transformer (Feature Tokenizer Transformer) for Tabular Data

Based on: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)

Key Innovation:
- ALL features (numerical AND categorical) are tokenized into embeddings
- Self-attention learns interactions between ALL feature types
- [CLS] token aggregates global context for final prediction

Architecture Flow:
  1. Numerical features: Linear projection per feature (b_j + x_j * W_j)
  2. Categorical features: Learned embeddings
  3. Concatenate all feature tokens
  4. Prepend [CLS] token
  5. Transformer encoder processes full sequence
  6. [CLS] embedding → MLP head for prediction

This is superior to TabTransformer because:
- TabTransformer: Only categoricals get attention, numericals bypass it
- FT-Transformer: All features participate in self-attention
- Critical for CICIDS (100% numerical) and UNSW (mostly numerical)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Config
# ------------------------------
@dataclass
class TabularExpertConfig:
    d_model: int = 64           # embedding dimension for all features
    n_heads: int = 4            # multi-head attention heads
    n_layers: int = 3           # transformer encoder layers
    dropout: float = 0.1

    # classification
    clf_hidden: int = 64
    n_classes: int = 2          # binary (normal vs attack)

    # masked-feature modeling (self-supervised pretraining)
    mask_ratio: float = 0.2     # ratio of features to mask per sample

    # FT-Transformer always uses [CLS] token
    use_cls_token: bool = True  # Always True for FT-Transformer

    # feature dimensions
    n_num: int = 0              # number of numerical features
    cat_cardinalities: List[int] = None  # cardinalities for categorical features
    
    # FT-Transformer specific
    numerical_embedding_bias: bool = True  # Use bias in numerical tokenization


# ------------------------------
# FT-Transformer Feature Tokenizer
# ------------------------------
class FTFeatureTokenizer(nn.Module):
    """
    FT-Transformer tokenization: ALL features → embeddings
    
    Numerical features: Each feature x_j is projected via:
        token_j = b_j + x_j * W_j
        where W_j: (d_model,), b_j: (d_model,)
    
    Categorical features: Standard embedding lookup
    
    Returns:
        [CLS] token + all feature tokens: (B, 1 + n_features, d_model)
    """
    def __init__(self, n_num: int, cat_cardinalities: List[int], d_model: int,
                 use_bias: bool = True):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities) if cat_cardinalities else 0
        self.n_features = self.n_num + self.n_cat
        self.d_model = d_model
        self.use_bias = use_bias

        # === NUMERICAL FEATURE TOKENIZATION ===
        # Per-feature linear projection: token_j = b_j + x_j * W_j
        if self.n_num > 0:
            # Weight matrix: (n_num, d_model) - each row is W_j for feature j
            self.num_weight = nn.Parameter(torch.randn(n_num, d_model))
            if use_bias:
                # Bias vector: (n_num, d_model) - each row is b_j for feature j
                self.num_bias = nn.Parameter(torch.randn(n_num, d_model))
            else:
                self.register_parameter('num_bias', None)
            
            # Initialize weights
            nn.init.xavier_uniform_(self.num_weight)
            if use_bias:
                nn.init.zeros_(self.num_bias)

        # === CATEGORICAL FEATURE TOKENIZATION ===
        # Standard embedding lookup per categorical feature
        if self.n_cat > 0:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_model) 
                for cardinality in cat_cardinalities
            ])

        # === [CLS] TOKEN ===
        # Learnable token that aggregates global context
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # === [MASK] TOKEN (for masked feature modeling) ===
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def tokenize_numerical(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        Tokenize numerical features using FT-Transformer projection
        
        x_num: (B, n_num)
        Returns: (B, n_num, d_model)
        
        For each feature j: token_j = b_j + x_j * W_j
        Implemented efficiently using broadcasting:
          x_num: (B, n_num, 1)
          W: (n_num, d_model)
          Result: (B, n_num, d_model)
        """
        # x_num: (B, n_num) -> (B, n_num, 1)
        x_expanded = x_num.unsqueeze(-1)  # (B, n_num, 1)
        
        # Broadcast multiplication: (B, n_num, 1) * (n_num, d_model) -> (B, n_num, d_model)
        tokens = x_expanded * self.num_weight.unsqueeze(0)  # (B, n_num, d_model)
        
        if self.use_bias:
            tokens = tokens + self.num_bias.unsqueeze(0)  # (B, n_num, d_model)
        
        return tokens

    def tokenize_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Tokenize categorical features using embeddings
        
        x_cat: (B, n_cat) - long tensor with category indices
        Returns: (B, n_cat, d_model)
        """
        tokens = []
        for i in range(self.n_cat):
            token = self.cat_embeddings[i](x_cat[:, i])  # (B, d_model)
            tokens.append(token)
        
        if tokens:
            return torch.stack(tokens, dim=1)  # (B, n_cat, d_model)
        else:
            # Return empty tensor if no categorical features
            return torch.empty(x_cat.size(0), 0, self.d_model, device=x_cat.device)

    def forward(self, x_num: Optional[torch.Tensor] = None, 
                x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Tokenize all features and prepend [CLS] token
        
        x_num: (B, n_num) or None
        x_cat: (B, n_cat) or None
        
        Returns: (B, 1 + n_features, d_model)
                 [CLS, num_feat_1, ..., num_feat_n, cat_feat_1, ..., cat_feat_m]
        """
        if x_num is None and x_cat is None:
            raise ValueError("At least one of x_num or x_cat must be provided")
        
        # Determine batch size
        B = x_num.size(0) if x_num is not None else x_cat.size(0)
        
        feature_tokens = []
        
        # Tokenize numerical features
        if x_num is not None and self.n_num > 0:
            num_tokens = self.tokenize_numerical(x_num)  # (B, n_num, d_model)
            feature_tokens.append(num_tokens)
        
        # Tokenize categorical features
        if x_cat is not None and self.n_cat > 0:
            cat_tokens = self.tokenize_categorical(x_cat)  # (B, n_cat, d_model)
            feature_tokens.append(cat_tokens)
        
        # Concatenate all feature tokens
        if feature_tokens:
            all_tokens = torch.cat(feature_tokens, dim=1)  # (B, n_features, d_model)
        else:
            raise ValueError("No features to tokenize")
        
        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, all_tokens], dim=1)  # (B, 1 + n_features, d_model)
        
        return tokens

    def apply_mask(self, tokens: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """
        Replace masked feature tokens with [MASK] token
        
        tokens: (B, 1 + n_features, d_model) - includes [CLS] at position 0
        mask_indices: (B, n_features) - boolean mask (True = masked)
        
        Returns: (B, 1 + n_features, d_model) with masked features replaced
        """
        B = tokens.size(0)
        masked_tokens = tokens.clone()
        
        # Expand mask token for batch
        mask_token = self.mask_token.expand(B, 1, self.d_model)  # (B, 1, d_model)
        
        # Apply mask to feature positions (skip [CLS] at index 0)
        for b in range(B):
            # Get indices of masked features (add 1 to skip [CLS])
            masked_positions = torch.where(mask_indices[b])[0] + 1
            if len(masked_positions) > 0:
                masked_tokens[b, masked_positions] = mask_token[b, 0]
        
        return masked_tokens


# ------------------------------
# Backbone Transformer
# ------------------------------
class TransformerBackbone(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L, D]
        h = self.encoder(tokens)     # [B, L, D]
        return self.norm(h)          # [B, L, D]


# ------------------------------
# Heads (Masked-Feature + Classification)
# ------------------------------
class MaskedFeatureHead(nn.Module):
    """
    Tête de reconstruction des features masquées.
    - Pour chaque feature numérique: régression (prédit la valeur normalisée)
    - Pour chaque feature catégorielle: classification (logits)
    On applique la perte seulement sur les features masquées.
    """
    def __init__(self, d_model: int, n_num: int, cat_cardinalities: List[int]):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities) if cat_cardinalities else 0

        # Une tête par feature numérique (linéaire -> scalaire)
        self.num_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(self.n_num)])

        # Une tête par feature catégorielle (linéaire -> logits)
        self.cat_heads = nn.ModuleList([nn.Linear(d_model, c) for c in (cat_cardinalities or [])])

        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self,
                h_tokens: torch.Tensor,
                mask_idx: torch.Tensor,
                y_num: Optional[torch.Tensor],
                y_cat: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        h_tokens: [B, 1+n_features, D]  (sortie du backbone)
        mask_idx: [B, n_features]  (True = feature masquée)
        y_num: [B, n_num] valeurs cibles numériques normalisées
        y_cat: [B, n_cat] valeurs cibles catégorielles (entiers)
        Retourne: (loss, logs)
        """
        B = h_tokens.size(0)
        logs = {}
        total_loss = 0.0
        total_count = 0

        # On ignore le token CLS => features = 1..n_features
        # num features = indices 0..n_num-1 dans mask_idx
        for i in range(self.n_num):
            feature_pos = i + 1
            pred = self.num_heads[i](h_tokens[:, feature_pos, :]).squeeze(-1)  # [B]
            target = y_num[:, i]                                              # [B]
            mask = mask_idx[:, i]                                             # [B] bool
            if mask.any():
                l = self.mse(pred[mask], target[mask])                        # [n_mask]
                total_loss += l.mean()
                total_count += 1

        # catégorielles
        for j in range(self.n_cat):
            feature_pos = self.n_num + j + 1
            pred_logits = self.cat_heads[j](h_tokens[:, feature_pos, :])      # [B, Cj]
            target = y_cat[:, j]                                              # [B]
            mask = mask_idx[:, self.n_num + j]                                # [B]
            if mask.any():
                l = self.ce(pred_logits[mask], target[mask])                  # [n_mask]
                total_loss += l.mean()
                total_count += 1

        if total_count == 0:
            # Rien n'a été masqué (ou batch vide) — renvoyer 0
            loss = torch.tensor(0.0, device=h_tokens.device, requires_grad=True)
        else:
            loss = total_loss / total_count

        logs["mfm_loss"] = loss.item()
        return loss, logs


class ClassificationHead(nn.Module):
    """
    Tête de classification (binaire par défaut).
    On pool les tokens (CLS ou mean) et on sort des logits.
    """
    def __init__(self, d_model: int, hidden: int, n_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: [B, D]
        return self.mlp(pooled)  # [B, n_classes]


# ------------------------------
# FT-Transformer Expert (Complete Pipeline)
# ------------------------------
class TabularExpert(nn.Module):
    """
    FT-Transformer Expert for Tabular Data
    
    Architecture:
      1. FTFeatureTokenizer: ALL features → embeddings + [CLS]
      2. TransformerBackbone: Self-attention over all feature tokens
      3. MaskedFeatureHead: Self-supervised pretraining
      4. ClassificationHead: Supervised finetuning (uses [CLS] token only)
    
    Key Advantage over TabTransformer:
      - All features (numerical + categorical) participate in self-attention
      - Critical for datasets like CICIDS (100% numerical) and UNSW (mostly numerical)
    """
    def __init__(self, cfg: TabularExpertConfig):
        super().__init__()
        self.cfg = cfg
        n_num = cfg.n_num
        cat_cards = cfg.cat_cardinalities or []

        # FT-Transformer tokenizer (tokenizes ALL features)
        self.tokenizer = FTFeatureTokenizer(
            n_num=n_num,
            cat_cardinalities=cat_cards,
            d_model=cfg.d_model,
            use_bias=cfg.numerical_embedding_bias
        )

        # Transformer backbone (processes full sequence)
        self.backbone = TransformerBackbone(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout
        )

        # Masked feature modeling head (pretraining)
        self.mfm_head = MaskedFeatureHead(
            d_model=cfg.d_model,
            n_num=n_num,
            cat_cardinalities=cat_cards
        )

        # Classification head (finetuning)
        self.clf_head = ClassificationHead(
            d_model=cfg.d_model,
            hidden=cfg.clf_hidden,
            n_classes=cfg.n_classes
        )

    def _extract_cls_token(self, h_tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract [CLS] token embedding from transformer output
        
        h_tokens: (B, 1 + n_features, d_model)
        Returns: (B, d_model) - just the [CLS] embedding at position 0
        """
        return h_tokens[:, 0, :]  # [CLS] is always at index 0

    def _sample_mask(self, B: int, n_features: int, device) -> torch.Tensor:
        """
        Sample random mask for masked feature modeling
        
        Returns: (B, n_features) boolean tensor
                 True = masked, False = visible
        """
        k = max(1, int(self.cfg.mask_ratio * n_features))
        mask = torch.zeros(B, n_features, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(n_features, device=device)[:k]
            mask[b, idx] = True
        return mask

    # ---- Self-Supervised Pretraining: Masked Feature Modeling ----
    def forward_mfm(self,
                    X_num: Optional[torch.Tensor] = None,
                    X_cat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Masked Feature Modeling (self-supervised pretraining)
        
        Process:
          1. Tokenize all features
          2. Randomly mask some feature tokens
          3. Transformer processes masked sequence
          4. Predict masked feature values
        
        Args:
            X_num: (B, n_num) normalized numerical features
            X_cat: (B, n_cat) categorical feature indices
        
        Returns:
            (loss, logs) - reconstruction loss and metrics
        """
        # Tokenize all features + prepend [CLS]
        tokens = self.tokenizer(x_num=X_num, x_cat=X_cat)  # (B, 1 + n_features, d_model)
        B, L, D = tokens.shape
        n_features = L - 1  # Exclude [CLS]

        # Sample random mask over features (not [CLS])
        mask_idx = self._sample_mask(B, n_features, tokens.device)  # (B, n_features)
        
        # Apply mask to tokens
        tokens_masked = self.tokenizer.apply_mask(tokens, mask_idx)  # (B, 1 + n_features, d_model)

        # Pass through transformer
        h = self.backbone(tokens_masked)  # (B, 1 + n_features, d_model)

        # Prepare targets for reconstruction
        y_num = X_num if X_num is not None and self.cfg.n_num > 0 else None
        y_cat = X_cat if X_cat is not None and len(self.cfg.cat_cardinalities or []) > 0 else None

        # Compute reconstruction loss (only on masked features)
        loss, logs = self.mfm_head(h, mask_idx, y_num, y_cat)
        return loss, logs

    # ---- Supervised Finetuning: Classification ----
    def forward_classify(self,
                         X_num: Optional[torch.Tensor] = None,
                         X_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classification forward pass (supervised finetuning)
        
        Process:
          1. Tokenize all features
          2. Transformer processes full sequence
          3. Extract [CLS] token embedding
          4. MLP head predicts class logits
        
        Args:
            X_num: (B, n_num) normalized numerical features
            X_cat: (B, n_cat) categorical feature indices
        
        Returns:
            logits: (B, n_classes) - class prediction logits
        """
        # Tokenize all features + prepend [CLS]
        tokens = self.tokenizer(x_num=X_num, x_cat=X_cat)  # (B, 1 + n_features, d_model)
        
        # Pass through transformer
        h = self.backbone(tokens)  # (B, 1 + n_features, d_model)
        
        # Extract [CLS] token (aggregates global context)
        cls_embedding = self._extract_cls_token(h)  # (B, d_model)
        
        # Classification head
        logits = self.clf_head(cls_embedding)  # (B, n_classes)
        return logits


# ------------------------------
# Petit test local (shape check)
# ------------------------------
if __name__ == "__main__":
    import torch

    # Define data structure
    num_num = 4
    num_cat = 3
    cat_cardinalities = [10, 8, 6]

    # === Create config ===
    cfg = TabularExpertConfig(
        n_num=num_num,
        cat_cardinalities=cat_cardinalities,
        d_model=64,
        n_heads=4,
        n_layers=2,
        clf_hidden=128,
        n_classes=2,
        dropout=0.1
    )

    # === Init model ===
    model = TabularExpert(cfg)

    # === Dummy input ===
    B = 8
    X_num = torch.randn(B, num_num)
    X_cat = torch.stack([torch.randint(0, c, (B,)) for c in cat_cardinalities], dim=1)

    # === Forward pass (self-supervised masking test) ===
    mfm_loss, logs = model.forward_mfm(X_num, X_cat)
    print("Masked Feature Modeling loss:", mfm_loss.item())

    # === Forward pass (classification head test) ===
    logits = model.forward_classify(X_num, X_cat)
    print("Classifier output shape:", logits.shape)

