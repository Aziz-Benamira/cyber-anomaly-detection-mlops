"""
Tabular Expert (Schema-Agnostic) — Feature Tokenizer + Transformer backbone
+ Masked-Feature pretraining head + Classification head.

Idée clé:
- Chaque feature (colonne) devient un "token" = embedding(nom_colonne) + encodage(valeur)
- On passe la séquence de tokens dans un Transformer (permutation-invariant si on shufflait)
- On peut pré-entraîner en self-supervised (masked feature modeling)
- Puis fine-tuner une tête de classification (attack / normal)

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
    d_model: int = 64           # dimension des embeddings / modèle
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    # classification
    clf_hidden: int = 64
    n_classes: int = 2          # binaire par défaut (normal vs attack)

    # masked-feature
    mask_ratio: float = 0.2     # ratio de features à masquer par échantillon

    # pooling
    use_cls_token: bool = True  # True: utilise un token [CLS]; False: mean pooling

    # initialisation des emb. noms de colonnes (optionnel: via LLM/word2vec)
    init_feature_name_embed: Optional[torch.Tensor] = None  # [n_features, d_model]

    # dimensions/features
    n_num: int = 0              # nb de colonnes numériques
    cat_cardinalities: List[int] = None  # liste des cardinalités pour chaque colonne catégorielle


# ------------------------------
# Tokenizer
# ------------------------------
class FeatureTokenizer(nn.Module):
    """
    Transforme X_num et X_cat en une séquence de tokens:
      token_i = Proj( Emb(feature_name_i)  ||  ValueEncoder(val_i) )

    - Les features numériques utilisent un petit MLP linéaire (1->d_model)
    - Les features catégorielles utilisent un Embedding(cardinality, d_model)
    - Embedding des noms de colonnes (feature_emb) appris (ou initialisé)
    """
    def __init__(self, n_num: int, cat_cardinalities: List[int], d_model: int,
                 init_feature_name_embed: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities) if cat_cardinalities else 0
        self.n_features = self.n_num + self.n_cat
        self.d_model = d_model

        # Embedding pour le nom des features (taille = n_features)
        self.feature_emb = nn.Embedding(self.n_features, d_model)
        if init_feature_name_embed is not None:
            if init_feature_name_embed.shape != (self.n_features, d_model):
                raise ValueError("init_feature_name_embed must have shape [n_features, d_model]")
            with torch.no_grad():
                self.feature_emb.weight.copy_(init_feature_name_embed)

        # Encodeur de valeur numérique
        self.num_encoder = nn.Linear(1, d_model)

        # Embeddings pour valeurs catégorielles
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in (cat_cardinalities or [])]
        )

        # Proj pour fusionner [name_emb | value_emb] -> d_model
        self.proj = nn.Linear(2 * d_model, d_model)

        # Token spécial [CLS] (optionnel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Token valeur masquée (pour masked-feature)
        self.mask_value_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, X_num: Optional[torch.Tensor], X_cat: Optional[torch.Tensor],
                add_cls: bool = True) -> torch.Tensor:
        """
        X_num: [B, n_num] (float)
        X_cat: [B, n_cat] (long)
        return: tokens [B, (n_features + cls), d_model]
        """
        B = X_num.size(0) if X_num is not None else X_cat.size(0)
        tokens = []

        # Numériques
        if X_num is not None and self.n_num > 0:
            for i in range(self.n_num):
                # nom de feature
                col_idx = torch.tensor(i, device=X_num.device)
                col_emb = self.feature_emb(col_idx)             # [d_model]
                col_emb = col_emb.unsqueeze(0).expand(B, -1)    # [B, d_model]

                # valeur encodée
                val_emb = self.num_encoder(X_num[:, i].unsqueeze(-1))  # [B, d_model]

                fused = torch.cat([col_emb, val_emb], dim=-1)   # [B, 2*d_model]
                fused = self.proj(fused)                        # [B, d_model]
                tokens.append(fused)

        # Catégorielles
        if X_cat is not None and self.n_cat > 0:
            for j in range(self.n_cat):
                feat_idx = self.n_num + j
                col_idx = torch.tensor(feat_idx, device=X_cat.device)
                col_emb = self.feature_emb(col_idx).unsqueeze(0).expand(B, -1)  # [B, d_model]

                val_emb = self.cat_embs[j](X_cat[:, j])  # [B, d_model]
                fused = torch.cat([col_emb, val_emb], dim=-1)
                fused = self.proj(fused)
                tokens.append(fused)

        if not tokens:
            raise ValueError("No features provided to tokenizer.")

        tokens = torch.stack(tokens, dim=1)  # [B, n_features, d_model]

        if add_cls:
            cls = self.cls_token.expand(B, 1, self.d_model)  # [B,1,d_model]
            tokens = torch.cat([cls, tokens], dim=1)         # [B, 1+n_features, d_model]

        return tokens

    def apply_value_mask(self, tokens: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        """
        Remplace la "partie valeur" des tokens par un token [MASK] simulé.
        Ici, comme on a déjà fusionné name+value via self.proj, on simule en remplaçant
        le token entier par mask_value_token aux positions masquées (hors CLS).
        tokens: [B, 1+n_features, d_model]
        mask_idx: [B, n_features] booleen — True si on masque cette feature
        """
        if tokens.size(1) != mask_idx.size(1) + 1:
            raise ValueError("mask_idx should refer to feature positions excluding CLS.")
        B = tokens.size(0)
        masked = tokens.clone()
        mask_token = self.mask_value_token.expand(B, 1, self.d_model)  # [B,1,D]
        # positions features = 1..n_features (0 = CLS)
        for b in range(B):
            # indices (dans tokens) à remplacer (décalage +1 pour ignorer CLS)
            idxs = torch.nonzero(mask_idx[b], as_tuple=False).squeeze(-1) + 1
            if idxs.numel() > 0:
                masked[b, idxs] = mask_token[b, 0, :]

        return masked


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
# Expert complet
# ------------------------------
class TabularExpert(nn.Module):
    """
    Pipeline complet "expert tabulaire":
      - Tokenizer (nom + valeur)
      - Backbone Transformer
      - Tête masked-feature (pré-entrainement)
      - Tête classification (fine-tuning)
    """
    def __init__(self, cfg: TabularExpertConfig):
        super().__init__()
        self.cfg = cfg
        n_num = cfg.n_num
        cat_cards = cfg.cat_cardinalities or []

        self.tokenizer = FeatureTokenizer(
            n_num=n_num,
            cat_cardinalities=cat_cards,
            d_model=cfg.d_model,
            init_feature_name_embed=cfg.init_feature_name_embed
        )

        self.backbone = TransformerBackbone(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout
        )

        self.mfm_head = MaskedFeatureHead(
            d_model=cfg.d_model,
            n_num=n_num,
            cat_cardinalities=cat_cards
        )

        self.clf_head = ClassificationHead(
            d_model=cfg.d_model,
            hidden=cfg.clf_hidden,
            n_classes=cfg.n_classes
        )

    def _pool(self, h_tokens: torch.Tensor) -> torch.Tensor:
        """
        h_tokens: [B, 1+n_features, D]
        """
        if self.cfg.use_cls_token:
            # Token 0 = CLS
            return h_tokens[:, 0, :]
        else:
            # Mean pooling (on peut exclure CLS si présent)
            return h_tokens.mean(dim=1)

    def _sample_mask(self, B: int, n_features: int, device) -> torch.Tensor:
        """
        Retourne un mask booléen [B, n_features] avec ~mask_ratio True.
        """
        k = max(1, int(self.cfg.mask_ratio * n_features))
        mask = torch.zeros(B, n_features, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(n_features, device=device)[:k]
            mask[b, idx] = True
        return mask

    # ---- Pré-entrainement: masked-feature modeling ----
    def forward_mfm(self,
                    X_num: Optional[torch.Tensor],
                    X_cat: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        X_num: [B, n_num] (déjà normalisé)
        X_cat: [B, n_cat]

        Retourne (loss, logs)
        """
        tokens = self.tokenizer(X_num, X_cat, add_cls=self.cfg.use_cls_token)  # [B, 1+n_features, D]
        B, L, D = tokens.shape
        n_features = L - 1  # hors CLS

        # mask indices (sur features seulement)
        mask_idx = self._sample_mask(B, n_features, tokens.device)  # [B, n_features]
        tokens_masked = self.tokenizer.apply_value_mask(tokens, mask_idx)  # [B, 1+n_features, D]

        # encode
        h = self.backbone(tokens_masked)  # [B, 1+n_features, D]

        # préparer cibles pour tête MFM
        y_num = X_num if X_num is not None and self.cfg.n_num > 0 else None
        y_cat = X_cat if X_cat is not None and len(self.cfg.cat_cardinalities or []) > 0 else None

        loss, logs = self.mfm_head(h, mask_idx, y_num, y_cat)
        return loss, logs

    # ---- Fine-tuning: classification ----
    def forward_classify(self,
                         X_num: Optional[torch.Tensor],
                         X_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Retourne des logits [B, n_classes]
        """
        tokens = self.tokenizer(X_num, X_cat, add_cls=self.cfg.use_cls_token)
        h = self.backbone(tokens)     # [B, L, D]
        pooled = self._pool(h)        # [B, D]
        logits = self.clf_head(pooled)
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

