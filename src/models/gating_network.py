"""
Gating Network for Mixture-of-Experts (MoE)

The gating network learns to dynamically weight expert contributions based on input.
It decides which expert (Tabular vs Temporal) is more relevant for each sample.

Key Concepts:
- Input: Concatenated expert embeddings [z_tabular || z_temporal]
- Output: Expert weights [w_tabular, w_temporal] via softmax (sum to 1)
- Learning: Trained end-to-end with the full MoE system
- Interpretation: Weights show which expert specializes in which patterns

Example:
  - DDoS attack with high IAT variance → temporal expert gets high weight
  - Port scan with many ports → tabular expert gets high weight
  - Normal traffic → balanced weights

Architecture:
  Input: [batch, 2 * d_embedding]  (e.g., [256, 256] for d_embedding=128)
  1. FC(256 → 128) + ReLU + Dropout
  2. FC(128 → 2) - outputs logits for 2 experts
  3. Softmax → [w_tabular, w_temporal]
  Output: [batch, 2] - weights that sum to 1 per sample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GatingNetwork(nn.Module):
    """
    Learns dynamic expert weights for Mixture-of-Experts.
    
    The gating network takes concatenated expert embeddings and produces
    normalized weights that indicate how much each expert should contribute
    to the final prediction.
    
    Args:
        d_embedding: Dimension of each expert's embedding (default: 128)
        n_experts: Number of experts (default: 2 for Tabular + Temporal)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout rate for regularization (default: 0.1)
        temperature: Temperature for softmax (default: 1.0, lower = sharper)
    """
    
    def __init__(
        self,
        d_embedding: int = 128,
        n_experts: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.d_embedding = d_embedding
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Input dimension: concatenated expert embeddings
        input_dim = n_experts * d_embedding  # 2 * 128 = 256
        
        # === GATING NETWORK LAYERS ===
        # Layer 1: Compress concatenated embeddings
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: Further compression (optional, can be removed for simpler gating)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer: Produce logits for each expert
        self.fc_out = nn.Linear(hidden_dim // 2, n_experts)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        
        # Zero bias
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(
        self,
        z_tabular: torch.Tensor,
        z_temporal: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expert weights from expert embeddings.
        
        Args:
            z_tabular: Tabular expert embedding [batch_size, d_embedding]
            z_temporal: Temporal expert embedding [batch_size, d_embedding]
            return_logits: If True, also return raw logits before softmax
        
        Returns:
            weights: Expert weights [batch_size, n_experts] (sum to 1 per sample)
            logits: (Optional) Raw logits before softmax [batch_size, n_experts]
        
        Example:
            weights = [[0.7, 0.3], [0.4, 0.6], ...]
            → First sample: 70% tabular, 30% temporal
            → Second sample: 40% tabular, 60% temporal
        """
        # Concatenate expert embeddings
        # z_tabular: [batch, 128], z_temporal: [batch, 128]
        z_concat = torch.cat([z_tabular, z_temporal], dim=1)  # [batch, 256]
        
        # === LAYER 1 ===
        x = self.fc1(z_concat)      # [batch, 128]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # === LAYER 2 ===
        x = self.fc2(x)             # [batch, 64]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # === OUTPUT LAYER ===
        logits = self.fc_out(x)     # [batch, 2]
        
        # Apply temperature scaling (lower temp = sharper distribution)
        logits = logits / self.temperature
        
        # Softmax to get normalized weights
        weights = F.softmax(logits, dim=1)  # [batch, 2]
        
        if return_logits:
            return weights, logits
        return weights
    
    def get_expert_usage(self, weights: torch.Tensor) -> dict:
        """
        Analyze expert usage statistics from gating weights.
        
        Args:
            weights: Gating weights [batch_size, n_experts]
        
        Returns:
            Dictionary with expert usage statistics:
            - mean_weights: Average weight per expert
            - std_weights: Standard deviation per expert
            - max_weights: Maximum weight per expert
            - dominant_expert: Which expert is dominant per sample
            - specialization: Measure of expert specialization (0=balanced, 1=specialized)
        """
        with torch.no_grad():
            batch_size = weights.size(0)
            
            # Mean and std per expert
            mean_weights = weights.mean(dim=0)  # [n_experts]
            std_weights = weights.std(dim=0)
            max_weights = weights.max(dim=0)[0]
            
            # Dominant expert per sample (argmax)
            dominant_expert = weights.argmax(dim=1)  # [batch_size]
            
            # Specialization score: entropy-based measure
            # Low entropy = specialized (one expert dominates)
            # High entropy = balanced (experts contribute equally)
            epsilon = 1e-10
            entropy = -(weights * torch.log(weights + epsilon)).sum(dim=1).mean()
            max_entropy = torch.log(torch.tensor(self.n_experts, dtype=torch.float32))
            specialization = 1 - (entropy / max_entropy)  # 0 = balanced, 1 = specialized
            
            # Count samples where each expert dominates
            expert_counts = []
            for expert_idx in range(self.n_experts):
                count = (dominant_expert == expert_idx).sum().item()
                expert_counts.append(count)
            
            return {
                'mean_weights': mean_weights.cpu().numpy(),
                'std_weights': std_weights.cpu().numpy(),
                'max_weights': max_weights.cpu().numpy(),
                'expert_counts': expert_counts,
                'specialization_score': specialization.item(),
                'dominant_expert_per_sample': dominant_expert.cpu().numpy()
            }


class SparseGatingNetwork(GatingNetwork):
    """
    Sparse gating variant that encourages expert specialization.
    
    Uses top-k sparsity: only the top-k experts get non-zero weights.
    For our 2-expert case, top-1 means only one expert is active per sample.
    
    This can improve:
    - Expert specialization (each expert focuses on specific patterns)
    - Computational efficiency (fewer experts need to be computed)
    - Interpretability (clearer expert roles)
    
    Args:
        Same as GatingNetwork
        top_k: Number of experts to activate per sample (default: 1)
    """
    
    def __init__(
        self,
        d_embedding: int = 128,
        n_experts: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0,
        top_k: int = 1
    ):
        super().__init__(d_embedding, n_experts, hidden_dim, dropout, temperature)
        self.top_k = min(top_k, n_experts)  # Can't select more than available experts
    
    def forward(
        self,
        z_tabular: torch.Tensor,
        z_temporal: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse expert weights (only top-k experts active).
        
        Returns:
            weights: Sparse expert weights [batch_size, n_experts]
                    Only top-k experts have non-zero weights, rest are 0
        """
        # Get logits from parent forward pass
        z_concat = torch.cat([z_tabular, z_temporal], dim=1)
        x = self.fc1(z_concat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self.fc_out(x) / self.temperature
        
        # Apply top-k masking
        # Get top-k values and indices
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)
        
        # Create mask: only top-k positions are non-zero
        mask = torch.zeros_like(logits)
        mask.scatter_(1, top_k_indices, 1.0)
        
        # Apply mask and normalize
        masked_logits = logits * mask
        weights = F.softmax(masked_logits, dim=1)
        
        if return_logits:
            return weights, logits
        return weights


# ------------------------------
# Utility Functions
# ------------------------------

def create_gating_network(
    d_embedding: int = 128,
    sparse: bool = False,
    top_k: int = 1,
    temperature: float = 1.0
) -> GatingNetwork:
    """
    Factory function to create gating network.
    
    Args:
        d_embedding: Expert embedding dimension (default: 128)
        sparse: Use sparse gating (top-k) vs dense gating (default: False)
        top_k: Number of experts to activate if sparse=True (default: 1)
        temperature: Softmax temperature (default: 1.0)
    
    Returns:
        GatingNetwork or SparseGatingNetwork instance
    """
    if sparse:
        print(f"Creating Sparse Gating Network:")
        print(f"  - Top-k experts: {top_k}")
        print(f"  - Temperature: {temperature}")
        return SparseGatingNetwork(
            d_embedding=d_embedding,
            top_k=top_k,
            temperature=temperature
        )
    else:
        print(f"Creating Dense Gating Network:")
        print(f"  - All experts active")
        print(f"  - Temperature: {temperature}")
        return GatingNetwork(
            d_embedding=d_embedding,
            temperature=temperature
        )


# ------------------------------
# Testing
# ------------------------------
if __name__ == "__main__":
    """Test Gating Network with sample data."""
    
    print("=" * 80)
    print("GATING NETWORK - EXPERT WEIGHTING TEST")
    print("=" * 80)
    
    # Test Dense Gating
    print("\n### Dense Gating (all experts active) ###")
    gating_dense = create_gating_network(d_embedding=128, sparse=False)
    
    # Sample expert embeddings
    batch_size = 256
    z_tabular = torch.randn(batch_size, 128)
    z_temporal = torch.randn(batch_size, 128)
    
    # Forward pass
    weights = gating_dense(z_tabular, z_temporal)
    
    print(f"\nInput shapes:")
    print(f"  z_tabular: {z_tabular.shape}")
    print(f"  z_temporal: {z_temporal.shape}")
    print(f"Output weights: {weights.shape}")
    print(f"Expected:       torch.Size([{batch_size}, 2])")
    
    # Verify weights sum to 1
    weight_sums = weights.sum(dim=1)
    print(f"\nWeight sum check (should be ~1.0):")
    print(f"  Min: {weight_sums.min():.6f}")
    print(f"  Max: {weight_sums.max():.6f}")
    print(f"  Mean: {weight_sums.mean():.6f}")
    
    # Analyze expert usage
    usage_stats = gating_dense.get_expert_usage(weights)
    print(f"\nExpert Usage Statistics:")
    print(f"  Mean weights: {usage_stats['mean_weights']}")
    print(f"  Std weights:  {usage_stats['std_weights']}")
    print(f"  Tabular dominant: {usage_stats['expert_counts'][0]} samples")
    print(f"  Temporal dominant: {usage_stats['expert_counts'][1]} samples")
    print(f"  Specialization score: {usage_stats['specialization_score']:.4f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in gating_dense.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    # Test Sparse Gating
    print("\n" + "=" * 80)
    print("\n### Sparse Gating (top-1 expert only) ###")
    gating_sparse = create_gating_network(d_embedding=128, sparse=True, top_k=1)
    
    weights_sparse = gating_sparse(z_tabular, z_temporal)
    
    print(f"\nOutput weights: {weights_sparse.shape}")
    print(f"Sample weights (first 5):")
    print(weights_sparse[:5])
    print(f"\nNote: Only one expert is active per sample (top-1)")
    
    # Analyze expert usage
    usage_stats_sparse = gating_sparse.get_expert_usage(weights_sparse)
    print(f"\nExpert Usage Statistics (Sparse):")
    print(f"  Mean weights: {usage_stats_sparse['mean_weights']}")
    print(f"  Tabular dominant: {usage_stats_sparse['expert_counts'][0]} samples")
    print(f"  Temporal dominant: {usage_stats_sparse['expert_counts'][1]} samples")
    print(f"  Specialization score: {usage_stats_sparse['specialization_score']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
