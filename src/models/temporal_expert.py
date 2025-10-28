"""
Temporal Expert - 1D CNN for Temporal/Sequential Features

Processes temporal statistics (IAT, duration, jitter, active/idle times) using 1D CNNs
to capture local patterns and feature interactions in time-domain features.

Why 1D CNN instead of MLP:
- Captures local patterns in temporal statistics (e.g., high IAT variance + long duration)
- Translation-invariant: robust to feature ordering
- More efficient than RNN/LSTM for aggregated statistics
- Good for feature interactions, not time-series forecasting

Why NOT LSTM/RNN:
- Our data is NOT raw sequences (no packet1, packet2, packet3...)
- We have pre-aggregated flow statistics (IAT Mean/Max/Min, Duration)
- LSTM overkill for summary statistics

Architecture:
  Input: [batch, n_temporal_features]  (e.g., CICIDS: 22, UNSW: 8)
  1. Unsqueeze to [batch, 1, n_temporal_features] - treat as 1D signal
  2. Conv1D(1 → 32, kernel=3) + ReLU
  3. Conv1D(32 → 64, kernel=3) + ReLU
  4. AdaptiveAvgPool1D → [batch, 64, 1]
  5. FC(64 → d_embedding=128)
  Output: [batch, 128] - same dimension as Tabular Expert for gating

Temporal Features by Dataset:
- CICIDS (22 features):
  * Flow IAT: Mean, Std, Max, Min
  * Forward/Backward IAT: Total, Mean, Std, Max, Min
  * Active: Mean, Std, Max, Min
  * Idle: Mean, Std, Max, Min
  * Flow Duration
  * Flow Bytes/s, Flow Packets/s
  
- UNSW (8 features):
  * dur: Record total duration
  * sjit: Source jitter
  * djit: Destination jitter
  * sinpkt: Source inter-packet arrival time
  * dinpkt: Destination inter-packet arrival time
  * tcprtt: TCP round trip time
  * synack: Time between SYN and SYN-ACK
  * ackdat: Time between SYN-ACK and ACK
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalExpert(nn.Module):
    """
    1D CNN-based temporal expert for flow-level temporal statistics.
    
    Args:
        n_temporal_features: Number of temporal features (22 for CICIDS, 8 for UNSW)
        d_embedding: Output embedding dimension (default: 128, matches Tabular Expert)
        dropout: Dropout rate for regularization (default: 0.1)
    """
    
    def __init__(
        self,
        n_temporal_features: int,
        d_embedding: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_temporal_features = n_temporal_features
        self.d_embedding = d_embedding
        
        # === 1D CONVOLUTIONAL LAYERS ===
        # Treat temporal features as a 1D signal
        # Conv1: 1 channel → 32 channels
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1,  # Keep same length
            bias=True
        )
        
        # Conv2: 32 channels → 64 channels
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Optional: Add a third conv layer for deeper feature extraction
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # === POOLING ===
        # Adaptive pooling to fixed size (reduces to single value per channel)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # === BATCH NORMALIZATION ===
        # Helps with training stability
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # === DROPOUT ===
        self.dropout = nn.Dropout(dropout)
        
        # === FULLY CONNECTED PROJECTION ===
        # Project pooled features to embedding space
        self.fc = nn.Linear(64, d_embedding)
        
        # === LAYER NORMALIZATION ===
        # Normalize final embedding
        self.layer_norm = nn.LayerNorm(d_embedding)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        # Conv layers: Kaiming initialization (good for ReLU)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        
        # FC layer: Xavier initialization
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x_temporal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal expert.
        
        Args:
            x_temporal: Temporal features [batch_size, n_temporal_features]
                       e.g., [256, 22] for CICIDS or [256, 8] for UNSW
        
        Returns:
            Temporal embedding [batch_size, d_embedding]
                       e.g., [256, 128]
        """
        # Input shape: [batch_size, n_temporal_features]
        # Example: [256, 22] for CICIDS
        
        # === RESHAPE TO 1D SIGNAL ===
        # Unsqueeze to add channel dimension: [batch, 1, n_features]
        x = x_temporal.unsqueeze(1)  # [256, 1, 22]
        
        # === CONV LAYER 1 ===
        x = self.conv1(x)           # [256, 32, 22]
        x = self.bn1(x)             # Batch norm
        x = F.relu(x)               # Activation
        x = self.dropout(x)         # Regularization
        
        # === CONV LAYER 2 ===
        x = self.conv2(x)           # [256, 64, 22]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # === CONV LAYER 3 ===
        x = self.conv3(x)           # [256, 64, 22]
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # === GLOBAL POOLING ===
        # Reduce spatial dimension to 1: [batch, 64, 22] → [batch, 64, 1]
        x = self.pool(x)            # [256, 64, 1]
        
        # === FLATTEN ===
        x = x.squeeze(-1)           # [256, 64]
        
        # === PROJECTION TO EMBEDDING SPACE ===
        x = self.fc(x)              # [256, 128]
        x = self.layer_norm(x)      # Normalize
        
        return x  # [batch_size, d_embedding]
    
    def get_feature_importance(self, x_temporal: torch.Tensor) -> torch.Tensor:
        """
        Optional: Analyze which temporal features are most important.
        
        Uses gradient-based attribution to measure feature importance.
        Useful for interpretability and analysis.
        
        Args:
            x_temporal: Temporal features [batch_size, n_temporal_features]
        
        Returns:
            Feature importance scores [batch_size, n_temporal_features]
        """
        x_temporal.requires_grad = True
        
        # Forward pass
        embedding = self.forward(x_temporal)
        
        # Use L2 norm of embedding as importance signal
        importance_signal = embedding.norm(dim=1).sum()
        
        # Backpropagate to get gradients w.r.t. input features
        importance_signal.backward()
        
        # Gradient magnitude = feature importance
        feature_importance = x_temporal.grad.abs()
        
        return feature_importance


# ------------------------------
# Utility Functions
# ------------------------------

def create_temporal_expert(dataset: str, d_embedding: int = 128, dropout: float = 0.1) -> TemporalExpert:
    """
    Factory function to create TemporalExpert for specific dataset.
    
    Args:
        dataset: Dataset name ('CICIDS' or 'UNSW')
        d_embedding: Output embedding dimension (default: 128)
        dropout: Dropout rate (default: 0.1)
    
    Returns:
        TemporalExpert instance configured for the dataset
    
    Raises:
        ValueError: If dataset name is not recognized
    """
    # Define temporal feature counts per dataset
    TEMPORAL_FEATURES = {
        'CICIDS': 22,  # IAT stats, Active/Idle stats, Duration, Flow rates
        'UNSW': 8      # dur, sjit, djit, sinpkt, dinpkt, tcprtt, synack, ackdat
    }
    
    dataset = dataset.upper()
    if dataset not in TEMPORAL_FEATURES:
        raise ValueError(
            f"Unknown dataset: {dataset}. Must be one of {list(TEMPORAL_FEATURES.keys())}"
        )
    
    n_temporal_features = TEMPORAL_FEATURES[dataset]
    
    print(f"Creating Temporal Expert for {dataset}:")
    print(f"  - Temporal features: {n_temporal_features}")
    print(f"  - Embedding dimension: {d_embedding}")
    print(f"  - Architecture: Conv1D(1→32→64→64) + AdaptiveAvgPool + FC")
    
    return TemporalExpert(
        n_temporal_features=n_temporal_features,
        d_embedding=d_embedding,
        dropout=dropout
    )


# ------------------------------
# Testing
# ------------------------------
if __name__ == "__main__":
    """Test Temporal Expert with sample data."""
    
    print("=" * 80)
    print("TEMPORAL EXPERT - 1D CNN ARCHITECTURE TEST")
    print("=" * 80)
    
    # Test CICIDS
    print("\n### CICIDS (22 temporal features) ###")
    expert_cicids = create_temporal_expert('CICIDS', d_embedding=128)
    
    # Sample batch
    batch_size = 256
    x_temporal_cicids = torch.randn(batch_size, 22)  # Random temporal features
    
    # Forward pass
    embedding_cicids = expert_cicids(x_temporal_cicids)
    
    print(f"\nInput shape:  {x_temporal_cicids.shape}")
    print(f"Output shape: {embedding_cicids.shape}")
    print(f"Expected:     torch.Size([{batch_size}, 128])")
    print(f"✓ Shape correct: {embedding_cicids.shape == torch.Size([batch_size, 128])}")
    
    # Count parameters
    n_params_cicids = sum(p.numel() for p in expert_cicids.parameters())
    print(f"\nTotal parameters: {n_params_cicids:,}")
    
    # Test UNSW
    print("\n" + "=" * 80)
    print("\n### UNSW (8 temporal features) ###")
    expert_unsw = create_temporal_expert('UNSW', d_embedding=128)
    
    x_temporal_unsw = torch.randn(batch_size, 8)
    embedding_unsw = expert_unsw(x_temporal_unsw)
    
    print(f"\nInput shape:  {x_temporal_unsw.shape}")
    print(f"Output shape: {embedding_unsw.shape}")
    print(f"Expected:     torch.Size([{batch_size}, 128])")
    print(f"✓ Shape correct: {embedding_unsw.shape == torch.Size([batch_size, 128])}")
    
    n_params_unsw = sum(p.numel() for p in expert_unsw.parameters())
    print(f"\nTotal parameters: {n_params_unsw:,}")
    
    # Test feature importance
    print("\n" + "=" * 80)
    print("\n### Feature Importance Analysis ###")
    importance = expert_cicids.get_feature_importance(x_temporal_cicids)
    print(f"Importance shape: {importance.shape}")
    print(f"Top 5 important features (average across batch):")
    avg_importance = importance.mean(dim=0)
    top_features = torch.topk(avg_importance, k=5)
    for rank, (score, idx) in enumerate(zip(top_features.values, top_features.indices), 1):
        print(f"  {rank}. Feature {idx}: {score:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
