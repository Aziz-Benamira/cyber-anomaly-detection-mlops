"""
MoE Inference Pipeline
======================

This module provides production-ready inference for the Mixture-of-Experts model.

Features:
1. Single sample prediction
2. Batch prediction
3. Feature importance analysis
4. Gating weight interpretation
5. Missing feature handling
6. Real-time inference optimization
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import warnings

from src.models.moe_model import create_moe_model, load_moe_model
from src.data.feature_config import get_feature_split


class MoEInference:
    """
    Production inference class for MoE models.
    
    Usage:
        # Initialize
        model = MoEInference(dataset='CICIDS', model_path='models/weights/cicids_moe_best.pt')
        
        # Single prediction
        result = model.predict(features_dict)
        
        # Batch prediction
        results = model.predict_batch(features_df)
        
        # With missing features
        result = model.predict(features_dict, handle_missing=True)
    """
    
    def __init__(
        self,
        dataset: str,
        model_path: str,
        metadata_path: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Initialize MoE inference pipeline.
        
        Args:
            dataset: "CICIDS" or "UNSW"
            model_path: Path to trained MoE checkpoint
            metadata_path: Path to dataset metadata (optional, auto-detected)
            device: 'auto', 'cuda', or 'cpu'
        """
        self.dataset = dataset
        self.model_path = Path(model_path)
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[INFO] Initializing MoE Inference for {dataset}")
        print(f"       Device: {self.device}")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = f"data/processed/{dataset.lower()}/metadata.json"
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract feature information
        self.feature_split = self.metadata.get('feature_split', {})
        self.n_tabular = self.feature_split.get('n_tabular', 0)
        self.n_temporal = self.feature_split.get('n_temporal', 0)
        self.temporal_names = self.feature_split.get('temporal_names', [])
        self.tabular_indices = self.feature_split.get('tabular_indices', [])
        self.temporal_indices = self.feature_split.get('temporal_indices', [])
        
        # Get all feature names
        self.all_feature_names = self._load_feature_names()
        self.tabular_feature_names = [self.all_feature_names[i] for i in self.tabular_indices]
        
        print(f"       Features: {self.n_tabular} tabular + {self.n_temporal} temporal")
        
        # Load model
        self.model = load_moe_model(
            dataset=dataset,
            n_tabular=self.n_tabular,
            n_temporal=self.n_temporal,
            model_path=str(self.model_path),
            device=self.device
        )
        self.model.eval()
        
        # Load preprocessing statistics (for normalization)
        self.scaler_stats = self._load_scaler_stats()
        
        print(f"[INFO] ✓ Model loaded and ready for inference!")
    
    def _load_feature_names(self) -> List[str]:
        """Load feature names from original dataset."""
        if self.dataset == "CICIDS":
            # CICIDS feature names
            return [
                'Destination Port', 'Flow Duration', 'Total Fwd Packets',
                'Total Backward Packets', 'Total Length of Fwd Packets',
                'Total Length of Bwd Packets', 'Fwd Packet Length Max',
                'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                'Fwd Packet Length Std', 'Bwd Packet Length Max',
                'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
                'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
                'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
                'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
                'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
                'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
                'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
                'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
                'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
                'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
                'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
                'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                'Active Mean', 'Active Std', 'Active Max', 'Active Min',
                'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
            ]
        else:  # UNSW
            # UNSW feature names (before one-hot encoding)
            return [
                'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
                'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
                'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
                'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
                'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                'response_body_len', 'ct_srv_src', 'ct_state_ttl',
                'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
                'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
            ]
    
    def _load_scaler_stats(self) -> Dict:
        """Load normalization statistics used during training."""
        try:
            scaler_path = f"data/processed/{self.dataset.lower()}/scaler_stats.json"
            if Path(scaler_path).exists():
                with open(scaler_path, 'r') as f:
                    return json.load(f)
        except:
            pass
        
        # Default: no scaling (model handles raw features)
        return {'mean': None, 'std': None}
    
    def preprocess_features(
        self,
        features: Union[Dict, pd.Series, np.ndarray],
        handle_missing: bool = False,
        missing_strategy: str = 'zero'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess raw features for model input.
        
        Args:
            features: Raw feature values (dict, pandas Series, or numpy array)
            handle_missing: Whether to handle missing features
            missing_strategy: How to fill missing values ('zero', 'mean', 'median')
        
        Returns:
            x_tabular: Preprocessed tabular features [1, n_tabular]
            x_temporal: Preprocessed temporal features [1, n_temporal]
        """
        # Convert to numpy array
        if isinstance(features, dict):
            # Dictionary: {feature_name: value}
            feature_array = np.zeros(len(self.all_feature_names))
            for i, name in enumerate(self.all_feature_names):
                if name in features:
                    feature_array[i] = features[name]
                elif handle_missing:
                    feature_array[i] = self._get_missing_value(name, missing_strategy)
                else:
                    raise ValueError(f"Missing required feature: {name}")
        
        elif isinstance(features, pd.Series):
            # Pandas Series
            feature_array = np.zeros(len(self.all_feature_names))
            for i, name in enumerate(self.all_feature_names):
                if name in features.index:
                    feature_array[i] = features[name]
                elif handle_missing:
                    feature_array[i] = self._get_missing_value(name, missing_strategy)
                else:
                    raise ValueError(f"Missing required feature: {name}")
        
        elif isinstance(features, np.ndarray):
            # Numpy array (assumed to be in correct order)
            if len(features) != len(self.all_feature_names):
                if handle_missing:
                    # Pad with missing values
                    feature_array = np.zeros(len(self.all_feature_names))
                    feature_array[:len(features)] = features
                else:
                    raise ValueError(f"Expected {len(self.all_feature_names)} features, got {len(features)}")
            else:
                feature_array = features
        
        else:
            raise TypeError(f"Unsupported feature type: {type(features)}")
        
        # Check for NaN/Inf
        if handle_missing:
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split into tabular and temporal
        x_tabular = feature_array[self.tabular_indices]
        x_temporal = feature_array[self.temporal_indices]
        
        # Convert to tensors
        x_tabular = torch.FloatTensor(x_tabular).unsqueeze(0).to(self.device)  # [1, n_tabular]
        x_temporal = torch.FloatTensor(x_temporal).unsqueeze(0).to(self.device)  # [1, n_temporal]
        
        # Handle categorical features for UNSW (one-hot encoded)
        if self.dataset == "UNSW":
            # UNSW features are already one-hot encoded in preprocessing
            x_tabular = x_tabular.long()  # Convert to indices for embedding
        
        return x_tabular, x_temporal
    
    def _get_missing_value(self, feature_name: str, strategy: str) -> float:
        """
        Get fill value for missing feature.
        
        Strategies:
        - 'zero': Fill with 0 (default)
        - 'mean': Fill with feature mean from training data
        - 'median': Fill with feature median from training data
        """
        if strategy == 'zero':
            return 0.0
        elif strategy == 'mean' and self.scaler_stats.get('mean') is not None:
            feature_idx = self.all_feature_names.index(feature_name)
            return self.scaler_stats['mean'][feature_idx]
        elif strategy == 'median' and self.scaler_stats.get('median') is not None:
            feature_idx = self.all_feature_names.index(feature_name)
            return self.scaler_stats['median'][feature_idx]
        else:
            return 0.0
    
    @torch.no_grad()
    def predict(
        self,
        features: Union[Dict, pd.Series, np.ndarray],
        handle_missing: bool = False,
        missing_strategy: str = 'zero',
        return_details: bool = True
    ) -> Dict:
        """
        Predict on a single sample.
        
        Args:
            features: Raw feature values
            handle_missing: Whether to handle missing features
            missing_strategy: How to fill missing values
            return_details: Whether to return detailed outputs (gating weights, etc.)
        
        Returns:
            Dictionary with:
            - prediction: 'Normal' or 'Attack'
            - confidence: Prediction confidence [0-1]
            - probabilities: [prob_normal, prob_attack]
            - logits: Raw model outputs
            - gating_weights: Expert weights [w_tabular, w_temporal]
            - expert_embeddings: Individual expert outputs (if return_details=True)
        
        Example:
            >>> model = MoEInference('CICIDS', 'models/weights/cicids_moe_best.pt')
            >>> features = {
            ...     'Flow Duration': 120000,
            ...     'Total Fwd Packets': 5,
            ...     'Flow IAT Mean': 24000,
            ...     # ... other features
            ... }
            >>> result = model.predict(features)
            >>> print(result['prediction'])  # 'Normal' or 'Attack'
            >>> print(result['confidence'])  # 0.987
            >>> print(result['gating_weights'])  # [0.52, 0.48]
        """
        # Preprocess features
        x_tabular, x_temporal = self.preprocess_features(
            features, handle_missing, missing_strategy
        )
        
        # Forward pass
        if return_details:
            # Get detailed outputs
            if self.dataset == "CICIDS":
                logits, z_tabular, z_temporal, gating_weights, z_combined = self.model(
                    x_tabular, None, x_temporal, return_expert_outputs=True
                )
            else:  # UNSW
                logits, z_tabular, z_temporal, gating_weights, z_combined = self.model(
                    None, x_tabular, x_temporal, return_expert_outputs=True
                )
        else:
            # Just get logits
            if self.dataset == "CICIDS":
                logits = self.model(x_tabular, None, x_temporal)
            else:  # UNSW
                logits = self.model(None, x_tabular, x_temporal)
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [prob_normal, prob_attack]
        
        # Get prediction
        pred_class = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        prediction = 'Normal' if pred_class == 0 else 'Attack'
        confidence = float(probs[pred_class])
        
        # Build result dictionary
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'Normal': float(probs[0]),
                'Attack': float(probs[1])
            },
            'logits': logits.cpu().numpy()[0].tolist()
        }
        
        if return_details:
            result['gating_weights'] = {
                'Tabular Expert': float(gating_weights[0, 0].cpu().numpy()),
                'Temporal Expert': float(gating_weights[0, 1].cpu().numpy())
            }
            result['expert_embeddings'] = {
                'tabular': z_tabular.cpu().numpy()[0].tolist(),
                'temporal': z_temporal.cpu().numpy()[0].tolist(),
                'combined': z_combined.cpu().numpy()[0].tolist()
            }
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        features_df: pd.DataFrame,
        handle_missing: bool = False,
        missing_strategy: str = 'zero',
        batch_size: int = 256
    ) -> pd.DataFrame:
        """
        Predict on a batch of samples (more efficient than individual predictions).
        
        Args:
            features_df: DataFrame with samples as rows, features as columns
            handle_missing: Whether to handle missing features
            missing_strategy: How to fill missing values
            batch_size: Batch size for processing
        
        Returns:
            DataFrame with predictions and confidence scores
        
        Example:
            >>> import pandas as pd
            >>> df = pd.read_csv('test_data.csv')
            >>> results = model.predict_batch(df)
            >>> print(results[['prediction', 'confidence', 'gating_tabular', 'gating_temporal']])
        """
        predictions = []
        confidences = []
        probs_normal = []
        probs_attack = []
        gating_tabular = []
        gating_temporal = []
        
        n_samples = len(features_df)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_df = features_df.iloc[start_idx:end_idx]
            
            # Preprocess batch
            batch_tabular = []
            batch_temporal = []
            
            for _, row in batch_df.iterrows():
                x_tab, x_temp = self.preprocess_features(row, handle_missing, missing_strategy)
                batch_tabular.append(x_tab)
                batch_temporal.append(x_temp)
            
            # Stack into batch tensors
            x_tabular_batch = torch.cat(batch_tabular, dim=0)  # [batch, n_tabular]
            x_temporal_batch = torch.cat(batch_temporal, dim=0)  # [batch, n_temporal]
            
            # Forward pass
            if self.dataset == "CICIDS":
                logits, _, _, gating_weights, _ = self.model(
                    x_tabular_batch, None, x_temporal_batch, return_expert_outputs=True
                )
            else:  # UNSW
                logits, _, _, gating_weights, _ = self.model(
                    None, x_tabular_batch, x_temporal_batch, return_expert_outputs=True
                )
            
            # Process outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
            
            for j in range(len(batch_df)):
                pred_class = int(pred_classes[j])
                predictions.append('Normal' if pred_class == 0 else 'Attack')
                confidences.append(float(probs[j, pred_class]))
                probs_normal.append(float(probs[j, 0]))
                probs_attack.append(float(probs[j, 1]))
                gating_tabular.append(float(gating_weights[j, 0].cpu().numpy()))
                gating_temporal.append(float(gating_weights[j, 1].cpu().numpy()))
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'prediction': predictions,
            'confidence': confidences,
            'prob_normal': probs_normal,
            'prob_attack': probs_attack,
            'gating_tabular': gating_tabular,
            'gating_temporal': gating_temporal
        })
        
        return results_df
    
    def analyze_prediction(self, features: Union[Dict, pd.Series, np.ndarray]) -> Dict:
        """
        Detailed analysis of a prediction (for debugging/interpretation).
        
        Returns:
            - Which expert was trusted more
            - Why (based on feature values)
            - Temporal vs tabular feature contributions
        """
        result = self.predict(features, return_details=True)
        
        # Analyze gating weights
        w_tabular = result['gating_weights']['Tabular Expert']
        w_temporal = result['gating_weights']['Temporal Expert']
        
        dominant_expert = 'Tabular' if w_tabular > w_temporal else 'Temporal'
        weight_diff = abs(w_tabular - w_temporal)
        
        analysis = {
            'prediction_summary': result['prediction'],
            'confidence': result['confidence'],
            'dominant_expert': dominant_expert,
            'expert_balance': 'Balanced' if weight_diff < 0.2 else 'Specialized',
            'gating_analysis': {
                'tabular_weight': w_tabular,
                'temporal_weight': w_temporal,
                'weight_difference': weight_diff,
                'interpretation': self._interpret_gating(w_tabular, w_temporal, result['prediction'])
            }
        }
        
        return analysis
    
    def _interpret_gating(self, w_tabular: float, w_temporal: float, prediction: str) -> str:
        """Generate human-readable interpretation of gating weights."""
        if w_temporal > 0.7:
            return (f"Model strongly relies on TEMPORAL features (timing patterns). "
                   f"This suggests the {prediction.lower()} classification is primarily based on "
                   f"timing anomalies (IAT, flow duration, packet rates).")
        elif w_tabular > 0.7:
            return (f"Model strongly relies on TABULAR features (protocol/port patterns). "
                   f"This suggests the {prediction.lower()} classification is primarily based on "
                   f"port combinations, flags, or packet statistics.")
        else:
            return (f"Model uses BALANCED combination of both experts. "
                   f"The {prediction.lower()} classification is based on both timing and protocol patterns.")


def demonstrate_inference():
    """
    Demonstration of MoE inference capabilities.
    """
    print("\n" + "="*80)
    print("MoE INFERENCE DEMONSTRATION")
    print("="*80)
    
    # Initialize model
    model = MoEInference(
        dataset='CICIDS',
        model_path='models/weights/cicids_moe_best.pt'
    )
    
    print("\n" + "-"*80)
    print("EXAMPLE 1: Normal Traffic Sample")
    print("-"*80)
    
    # Example normal traffic features (realistic values)
    normal_features = {
        'Destination Port': 80,
        'Flow Duration': 5000,
        'Total Fwd Packets': 3,
        'Total Backward Packets': 2,
        'Flow IAT Mean': 1500,
        'Flow IAT Std': 200,
        'SYN Flag Count': 1,
        'ACK Flag Count': 4,
        # ... (other features would be here)
    }
    
    result = model.predict(normal_features, handle_missing=True)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Gating Weights:")
    print(f"  - Tabular Expert: {result['gating_weights']['Tabular Expert']:.3f}")
    print(f"  - Temporal Expert: {result['gating_weights']['Temporal Expert']:.3f}")
    
    # Detailed analysis
    analysis = model.analyze_prediction(normal_features)
    print(f"\nInterpretation: {analysis['gating_analysis']['interpretation']}")
    
    print("\n" + "-"*80)
    print("EXAMPLE 2: DDoS Attack Sample (with missing features)")
    print("-"*80)
    
    # Example DDoS attack (only some features available)
    ddos_features = {
        'Flow Duration': 1000,      # Very short
        'Total Fwd Packets': 100,   # Many packets
        'Flow IAT Mean': 10,         # Very low IAT (rapid packets)
        'Flow IAT Std': 5,           # Low variance
        'SYN Flag Count': 100,       # SYN flood
        # Missing: Many other features (simulating real-time scenario)
    }
    
    result = model.predict(ddos_features, handle_missing=True)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Gating Weights:")
    print(f"  - Tabular Expert: {result['gating_weights']['Tabular Expert']:.3f}")
    print(f"  - Temporal Expert: {result['gating_weights']['Temporal Expert']:.3f}")
    
    # Detailed analysis
    analysis = model.analyze_prediction(ddos_features)
    print(f"\nInterpretation: {analysis['gating_analysis']['interpretation']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demonstrate_inference()
