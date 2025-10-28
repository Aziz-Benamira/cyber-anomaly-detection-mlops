"""
Mixture-of-Experts (MoE) Model for Cybersecurity Anomaly Detection

Combines multiple specialized experts (Tabular + Temporal) with a gating network
to dynamically weight their contributions for each input sample.

Architecture:
    Input: x_tabular (non-temporal features), x_temporal (timing features)
    
    1. Tabular Expert (FT-Transformer):
       - Processes non-temporal features (flags, packet counts, protocols, etc.)
       - Pre-trained with Masked Feature Modeling
       - Outputs: z_tabular [batch, d_embedding]
    
    2. Temporal Expert (1D CNN):
       - Processes temporal features (IAT, duration, jitter, active/idle)
       - 1D CNNs capture local patterns in time-domain statistics
       - Outputs: z_temporal [batch, d_embedding]
    
    3. Gating Network:
       - Takes [z_tabular || z_temporal] as input
       - Learns dynamic weights [w_tabular, w_temporal] via softmax
       - Outputs: weights [batch, 2]
    
    4. Weighted Combination:
       - z_final = w_tabular * z_tabular + w_temporal * z_temporal
    
    5. Classifier:
       - MLP head: z_final → [Normal, Attack]

Benefits over Single Expert:
- Expert Specialization: Each expert focuses on its strength
  * Tabular: Protocol patterns, flag combinations, port scans
  * Temporal: Timing anomalies, DDoS bursts, slow scans
- Dynamic Weighting: Gating network adapts per sample
  * DDoS → high temporal weight (timing patterns important)
  * Port Scan → high tabular weight (port/protocol patterns important)
- Improved Generalization: Combines complementary information

Feature Split by Dataset:
- CICIDS: 50 tabular + 22 temporal features
- UNSW: 31 tabular + 8 temporal features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .tabular_expert import TabularExpert, TabularExpertConfig
from .temporal_expert import TemporalExpert
from .gating_network import GatingNetwork


class MixtureOfExperts(nn.Module):
    """
    2-Expert MoE model combining Tabular (FT-Transformer) and Temporal (1D CNN) experts.
    
    Args:
        tabular_expert: Pre-initialized TabularExpert (can load pretrained weights)
        temporal_expert: TemporalExpert for temporal features
        gating_network: GatingNetwork for expert weighting
        d_embedding: Embedding dimension (must match expert outputs, default: 128)
        n_classes: Number of output classes (default: 2 for binary classification)
        dropout: Dropout rate for classifier (default: 0.1)
    """
    
    def __init__(
        self,
        tabular_expert: TabularExpert,
        temporal_expert: TemporalExpert,
        gating_network: GatingNetwork,
        d_embedding: int = 128,
        n_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_embedding = d_embedding
        self.n_classes = n_classes
        
        # === EXPERTS ===
        self.tabular_expert = tabular_expert
        self.temporal_expert = temporal_expert
        
        # === GATING NETWORK ===
        self.gating_network = gating_network
        
        # === CLASSIFIER HEAD ===
        # Takes combined embedding and outputs class logits
        self.classifier = nn.Sequential(
            nn.Linear(d_embedding, d_embedding),
            nn.LayerNorm(d_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_embedding, n_classes)
        )
        
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_tabular_num: torch.Tensor,
        x_tabular_cat: Optional[torch.Tensor],
        x_temporal: torch.Tensor,
        return_expert_outputs: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through MoE model.
        
        Args:
            x_tabular_num: Numerical tabular features [batch, n_tabular_num]
            x_tabular_cat: Categorical tabular features [batch, n_tabular_cat] (can be None)
            x_temporal: Temporal features [batch, n_temporal]
            return_expert_outputs: If True, return expert embeddings and gating weights
        
        Returns:
            logits: Class prediction logits [batch, n_classes]
            
            If return_expert_outputs=True, also returns:
            - z_tabular: Tabular expert embedding [batch, d_embedding]
            - z_temporal: Temporal expert embedding [batch, d_embedding]
            - gating_weights: Expert weights [batch, 2]
            - z_combined: Final combined embedding [batch, d_embedding]
        """
        # === EXPERT FORWARD PASSES ===
        # Get embeddings from each expert (not final logits)
        
        # Tabular Expert: Get [CLS] token embedding from FT-Transformer
        tabular_tokens = self.tabular_expert.tokenizer(
            x_num=x_tabular_num,
            x_cat=x_tabular_cat
        )  # [batch, 1 + n_features, d_model]
        
        tabular_hidden = self.tabular_expert.backbone(tabular_tokens)  # [batch, 1 + n_features, d_model]
        z_tabular = self.tabular_expert._extract_cls_token(tabular_hidden)  # [batch, d_model]
        
        # Temporal Expert: Get embedding from 1D CNN
        z_temporal = self.temporal_expert(x_temporal)  # [batch, d_embedding]
        
        # === GATING NETWORK ===
        # Compute expert weights based on embeddings
        gating_weights = self.gating_network(z_tabular, z_temporal)  # [batch, 2]
        
        # === WEIGHTED COMBINATION ===
        # Combine expert embeddings using gating weights
        # z_combined = w_tabular * z_tabular + w_temporal * z_temporal
        w_tabular = gating_weights[:, 0:1]  # [batch, 1]
        w_temporal = gating_weights[:, 1:2]  # [batch, 1]
        
        z_combined = w_tabular * z_tabular + w_temporal * z_temporal  # [batch, d_embedding]
        
        # === CLASSIFICATION ===
        logits = self.classifier(z_combined)  # [batch, n_classes]
        
        if return_expert_outputs:
            return logits, z_tabular, z_temporal, gating_weights, z_combined
        
        return logits
    
    def get_expert_embeddings(
        self,
        x_tabular_num: torch.Tensor,
        x_tabular_cat: Optional[torch.Tensor],
        x_temporal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract expert embeddings without classification.
        Useful for visualization and analysis.
        
        Returns:
            z_tabular: Tabular expert embedding [batch, d_embedding]
            z_temporal: Temporal expert embedding [batch, d_embedding]
        """
        with torch.no_grad():
            # Tabular
            tabular_tokens = self.tabular_expert.tokenizer(x_num=x_tabular_num, x_cat=x_tabular_cat)
            tabular_hidden = self.tabular_expert.backbone(tabular_tokens)
            z_tabular = self.tabular_expert._extract_cls_token(tabular_hidden)
            
            # Temporal
            z_temporal = self.temporal_expert(x_temporal)
        
        return z_tabular, z_temporal
    
    def analyze_gating_weights(
        self,
        x_tabular_num: torch.Tensor,
        x_tabular_cat: Optional[torch.Tensor],
        x_temporal: torch.Tensor,
        y_true: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Analyze gating weights to understand expert specialization.
        
        Args:
            x_tabular_num: Numerical tabular features
            x_tabular_cat: Categorical tabular features
            x_temporal: Temporal features
            y_true: True labels (optional, for attack type analysis)
        
        Returns:
            Dictionary with gating statistics:
            - mean_weights: Average weight per expert
            - weights_by_class: Weights grouped by class (if y_true provided)
            - specialization_score: Measure of expert specialization
        """
        with torch.no_grad():
            # Get embeddings
            tabular_tokens = self.tabular_expert.tokenizer(x_num=x_tabular_num, x_cat=x_tabular_cat)
            tabular_hidden = self.tabular_expert.backbone(tabular_tokens)
            z_tabular = self.tabular_expert._extract_cls_token(tabular_hidden)
            z_temporal = self.temporal_expert(x_temporal)
            
            # Get gating weights
            gating_weights = self.gating_network(z_tabular, z_temporal)  # [batch, 2]
            
            # Compute statistics
            stats = self.gating_network.get_expert_usage(gating_weights)
            
            # If labels provided, analyze weights by class
            if y_true is not None:
                # Normal samples (class 0)
                normal_mask = (y_true == 0)
                if normal_mask.sum() > 0:
                    stats['normal_weights'] = gating_weights[normal_mask].mean(dim=0).cpu().numpy()
                
                # Attack samples (class 1)
                attack_mask = (y_true == 1)
                if attack_mask.sum() > 0:
                    stats['attack_weights'] = gating_weights[attack_mask].mean(dim=0).cpu().numpy()
            
            return stats
    
    def load_pretrained_tabular_expert(self, checkpoint_path: str):
        """
        Load pretrained weights for tabular expert (from FT-Transformer pretraining).
        
        This enables transfer learning: the tabular expert starts with good
        feature representations learned via Masked Feature Modeling.
        
        Args:
            checkpoint_path: Path to pretrained TabularExpert checkpoint
        """
        print(f"Loading pretrained Tabular Expert from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load only the tabular expert weights
        self.tabular_expert.load_state_dict(checkpoint, strict=False)
        print("✓ Pretrained Tabular Expert loaded successfully")
    
    def freeze_tabular_expert(self):
        """
        Freeze tabular expert weights (only train temporal expert + gating).
        
        Useful for:
        - Quick experiments with fixed tabular representations
        - When tabular expert is already well-trained
        - Reducing computational cost
        """
        for param in self.tabular_expert.parameters():
            param.requires_grad = False
        print("✓ Tabular Expert frozen")
    
    def unfreeze_all(self):
        """Unfreeze all experts for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ All components unfrozen")


# ------------------------------
# Factory Functions
# ------------------------------

def create_moe_model(
    dataset: str,
    n_tabular_num: int,
    n_tabular_cat: int = 0,
    cat_cardinalities: Optional[list] = None,
    d_embedding: int = 128,
    clf_hidden: int = 256,
    pretrained_tabular_path: Optional[str] = None,
    sparse_gating: bool = False,
    freeze_tabular: bool = False
) -> MixtureOfExperts:
    """
    Factory function to create MoE model for specific dataset.
    
    Args:
        dataset: Dataset name ('CICIDS' or 'UNSW')
        n_tabular_num: Number of numerical tabular features
        n_tabular_cat: Number of categorical tabular features (default: 0)
        cat_cardinalities: List of cardinalities for categorical features
        d_embedding: Embedding dimension (default: 128)
        clf_hidden: Hidden dimension for classification head (default: 256)
        pretrained_tabular_path: Path to pretrained TabularExpert weights (optional)
        sparse_gating: Use sparse (top-k) gating (default: False)
        freeze_tabular: Freeze tabular expert weights (default: False)
    
    Returns:
        MixtureOfExperts model
    """
    from .temporal_expert import create_temporal_expert
    from .gating_network import create_gating_network
    
    print("=" * 80)
    print(f"CREATING MIXTURE-OF-EXPERTS MODEL FOR {dataset}")
    print("=" * 80)
    
    # === TABULAR EXPERT ===
    print("\n### Tabular Expert (FT-Transformer) ###")
    tabular_config = TabularExpertConfig(
        n_num=n_tabular_num,
        cat_cardinalities=cat_cardinalities if cat_cardinalities else [],
        d_model=d_embedding,
        n_heads=4,
        n_layers=3,
        clf_hidden=clf_hidden,  # Use provided clf_hidden
        n_classes=2,
        dropout=0.1
    )
    tabular_expert = TabularExpert(tabular_config)
    print(f"  - Numerical features: {n_tabular_num}")
    print(f"  - Categorical features: {n_tabular_cat}")
    print(f"  - Embedding dimension: {d_embedding}")
    print(f"  - Classifier hidden: {clf_hidden}")
    
    # Load pretrained weights if provided
    if pretrained_tabular_path:
        checkpoint = torch.load(pretrained_tabular_path, map_location='cpu')
        tabular_expert.load_state_dict(checkpoint, strict=False)
        print(f"  ✓ Loaded pretrained weights from: {pretrained_tabular_path}")
    
    # === TEMPORAL EXPERT ===
    print("\n### Temporal Expert (1D CNN) ###")
    temporal_expert = create_temporal_expert(dataset, d_embedding=d_embedding)
    
    # === GATING NETWORK ===
    print("\n### Gating Network ###")
    gating_network = create_gating_network(
        d_embedding=d_embedding,
        sparse=sparse_gating,
        top_k=1 if sparse_gating else None
    )
    
    # === MOE MODEL ===
    print("\n### Assembling MoE ###")
    moe = MixtureOfExperts(
        tabular_expert=tabular_expert,
        temporal_expert=temporal_expert,
        gating_network=gating_network,
        d_embedding=d_embedding,
        n_classes=2
    )
    
    # Freeze tabular expert if requested
    if freeze_tabular:
        moe.freeze_tabular_expert()
    
    # Count total parameters
    total_params = sum(p.numel() for p in moe.parameters())
    trainable_params = sum(p.numel() for p in moe.parameters() if p.requires_grad)
    
    print(f"\n### Model Statistics ###")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Frozen: {freeze_tabular}")
    
    print("\n" + "=" * 80)
    print("✅ MoE MODEL CREATED SUCCESSFULLY")
    print("=" * 80)
    
    return moe


# ------------------------------
# Testing
# ------------------------------
if __name__ == "__main__":
    """Test MoE model with sample data."""
    
    print("\n" * 2)
    print("=" * 80)
    print("MIXTURE-OF-EXPERTS MODEL TEST")
    print("=" * 80)
    
    # Test CICIDS configuration
    print("\n### CICIDS Configuration ###")
    print("Tabular features: 50 numerical, 0 categorical")
    print("Temporal features: 22")
    
    moe_cicids = create_moe_model(
        dataset='CICIDS',
        n_tabular_num=50,
        n_tabular_cat=0,
        d_embedding=128,
        sparse_gating=False
    )
    
    # Sample batch
    batch_size = 256
    x_tabular_num = torch.randn(batch_size, 50)
    x_tabular_cat = None  # No categorical features in CICIDS
    x_temporal = torch.randn(batch_size, 22)
    
    # Forward pass
    print("\n### Forward Pass ###")
    logits, z_tab, z_temp, weights, z_combined = moe_cicids(
        x_tabular_num,
        x_tabular_cat,
        x_temporal,
        return_expert_outputs=True
    )
    
    print(f"Input shapes:")
    print(f"  x_tabular_num: {x_tabular_num.shape}")
    print(f"  x_temporal: {x_temporal.shape}")
    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  z_tabular: {z_tab.shape}")
    print(f"  z_temporal: {z_temp.shape}")
    print(f"  Gating weights: {weights.shape}")
    print(f"  z_combined: {z_combined.shape}")
    
    # Analyze gating weights
    print(f"\n### Gating Weights Analysis ###")
    print(f"Sample weights (first 5):")
    print(weights[:5])
    
    stats = moe_cicids.gating_network.get_expert_usage(weights)
    print(f"\nMean weights: {stats['mean_weights']}")
    print(f"  - Tabular: {stats['mean_weights'][0]:.3f}")
    print(f"  - Temporal: {stats['mean_weights'][1]:.3f}")
    print(f"Specialization score: {stats['specialization_score']:.3f}")
    
    # Test UNSW configuration
    print("\n" + "=" * 80)
    print("\n### UNSW Configuration ###")
    print("Tabular features: 28 numerical, 3 categorical")
    print("Temporal features: 8")
    
    moe_unsw = create_moe_model(
        dataset='UNSW',
        n_tabular_num=28,
        n_tabular_cat=3,
        cat_cardinalities=[3, 132, 13],  # proto, service, state
        d_embedding=128,
        sparse_gating=False
    )
    
    x_tabular_num_unsw = torch.randn(batch_size, 28)
    x_tabular_cat_unsw = torch.randint(0, 3, (batch_size, 3))
    x_temporal_unsw = torch.randn(batch_size, 8)
    
    logits_unsw = moe_unsw(x_tabular_num_unsw, x_tabular_cat_unsw, x_temporal_unsw)
    print(f"\nOutput logits: {logits_unsw.shape}")
    
    print("\n" + "=" * 80)


def load_moe_model(dataset: str, n_tabular: int, n_temporal: int, 
                   model_path: str, device: torch.device) -> MixtureOfExperts:
    """Load a trained MoE model from checkpoint.
    
    Args:
        dataset: "CICIDS" or "UNSW"
        n_tabular: Number of tabular features
        n_temporal: Number of temporal features
        model_path: Path to model checkpoint (.pt file)
        device: torch.device to load model on
    
    Returns:
        Loaded MoE model in eval mode
    """
    import yaml
    
    # Load params
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    
    # Get model config from params
    d_model = params['train']['d_model']
    clf_hidden = params['train']['clf_hidden']
    
    # Create model architecture (need to match training config)
    if dataset == "CICIDS":
        # CICIDS: Numerical features only
        moe = create_moe_model(
            dataset=dataset,
            n_tabular_num=n_tabular,
            n_tabular_cat=0,
            cat_cardinalities=[],
            d_embedding=d_model,
            clf_hidden=clf_hidden,
            sparse_gating=False
        )
    elif dataset == "UNSW":
        # UNSW: All categorical (one-hot encoded)
        moe = create_moe_model(
            dataset=dataset,
            n_tabular_num=0,
            n_tabular_cat=n_tabular,
            cat_cardinalities=[2] * n_tabular,  # All binary one-hot
            d_embedding=d_model,
            clf_hidden=clf_hidden,
            sparse_gating=False
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint is wrapped or raw state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        moe.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Raw state_dict
        moe.load_state_dict(checkpoint)
    
    moe.to(device)
    moe.eval()
    
    print(f"✓ Loaded MoE model from {model_path}")
    print(f"  Total parameters: {sum(p.numel() for p in moe.parameters()):,}")
    
    return moe
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
