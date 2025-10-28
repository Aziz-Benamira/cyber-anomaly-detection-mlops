"""
Train Mixture-of-Experts (MoE) Model for Cybersecurity Anomaly Detection

This script trains a 2-Expert MoE combining:
1. Tabular Expert (FT-Transformer) - initialized with pretrained weights
2. Temporal Expert (1D CNN) - trained from scratch
3. Gating Network - learns dynamic expert weighting
4. Classifier - final prediction layer

Key features:
- Load X_tabular.npy and X_temporal.npy separately
- Weighted loss for imbalanced data
- Comprehensive metrics (F1, Precision, Recall, AUC-PR)
- Log gating weights to MLflow for analysis
- Transfer learning from pretrained Tabular Expert

Usage:
    python -m src.models.train_moe --stage pretrain  # Pretrain Tabular Expert on tabular features only
    python -m src.models.train_moe --stage train     # Train full MoE end-to-end
"""

import os
import yaml
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score
)
import numpy as np
import json

from src.models.moe_model import create_moe_model
from src.models.tabular_expert import TabularExpert, TabularExpertConfig


# ============================================================
#  Class Weight Calculation
# ============================================================
def calculate_class_weights(y_tensor, device):
    """
    Calculate class weights for imbalanced datasets.
    
    Formula: weight_i = total_samples / (num_classes * count_i)
    """
    unique, counts = torch.unique(y_tensor, return_counts=True)
    total_samples = len(y_tensor)
    num_classes = len(unique)
    
    weights = torch.zeros(num_classes, device=device)
    for class_idx, count in zip(unique, counts):
        weights[class_idx] = total_samples / (num_classes * count)
    
    print(f"\n[INFO] Class Weight Calculation:")
    print(f"       Total samples: {total_samples:,}")
    print(f"       Class distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"         Class {class_idx}: {count:,} samples ({count/total_samples*100:.1f}%) -> weight={weights[class_idx]:.4f}")
    print(f"       Weight ratio (attack/normal): {weights[1]/weights[0]:.2f}x")
    print(f"       ⚠️  Minority class errors penalized {weights[1]/weights[0]:.2f}x more!\n")
    
    return weights


# ============================================================
#  Load MoE Dataset (Tabular + Temporal)
# ============================================================
def load_moe_dataset(dataset_dir, batch_size=256, test_size=0.2, random_state=42):
    """
    Load tabular and temporal features separately for MoE.
    
    Returns:
        train_loader, val_loader, metadata
    """
    # Load split features
    X_tabular = np.load(f"{dataset_dir}/X_tabular.npy", allow_pickle=True)
    X_temporal = np.load(f"{dataset_dir}/X_temporal.npy", allow_pickle=True)
    y = np.load(f"{dataset_dir}/y.npy", allow_pickle=True)
    
    # Load metadata
    with open(f"{dataset_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"[INFO] Loaded MoE dataset from {dataset_dir}")
    print(f"       X_tabular shape: {X_tabular.shape}")
    print(f"       X_temporal shape: {X_temporal.shape}")
    print(f"       y shape: {y.shape}")
    print(f"       Feature split: {metadata.get('feature_split', {})}")
    
    # Convert to correct dtypes
    X_tabular = X_tabular.astype(np.float32)
    X_temporal = X_temporal.astype(np.float32)
    y = y.astype(np.int64)
    
    # Check label distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"[INFO] Label distribution: {dict(zip(unique, counts))}")
    
    # Convert to tensors
    X_tabular = torch.tensor(X_tabular, dtype=torch.float32)
    X_temporal = torch.tensor(X_temporal, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Stratified split
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y.numpy()
    )
    
    print(f"[INFO] Stratified split: {len(train_idx)} train, {len(val_idx)} val")
    print(f"       Train labels: {np.bincount(y[train_idx].numpy())}")
    print(f"       Val labels:   {np.bincount(y[val_idx].numpy())}")
    
    # Create datasets with both tabular and temporal features
    ds = TensorDataset(X_tabular, X_temporal, y)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        metadata
    )


# ============================================================
#  Load Tabular-Only Dataset (for pretraining Tabular Expert)
# ============================================================
def load_tabular_only_dataset(dataset_dir, batch_size=256, test_size=0.2, random_state=42):
    """
    Load only tabular features for pretraining Tabular Expert.
    """
    X_tabular = np.load(f"{dataset_dir}/X_tabular.npy", allow_pickle=True)
    y = np.load(f"{dataset_dir}/y.npy", allow_pickle=True)
    
    print(f"[INFO] Loaded tabular-only dataset from {dataset_dir}")
    print(f"       X_tabular shape: {X_tabular.shape}")
    
    X_tabular = X_tabular.astype(np.float32)
    y = y.astype(np.int64)
    
    X_tabular = torch.tensor(X_tabular, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Stratified split
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y.numpy()
    )
    
    ds = TensorDataset(X_tabular, y)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    )


# ============================================================
#  Metrics Calculation
# ============================================================
def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics for imbalanced anomaly detection."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Per-class metrics (focus on attack class = 1)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # AUC metrics
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.0
    
    try:
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(recalls, precisions)
    except:
        auc_pr = 0.0
    
    # Overall accuracy (for reference, but NOT the main metric!)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_pr': auc_pr,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


# ============================================================
#  Pretrain Tabular Expert (Self-Supervised MFM)
# ============================================================
def pretrain_tabular_expert(model, dataloader, optimizer, device, epochs=1):
    """
    Pretrain Tabular Expert with Masked Feature Modeling.
    Only uses tabular features (no temporal).
    """
    print("\n" + "="*80)
    print("PRETRAINING TABULAR EXPERT (Masked Feature Modeling)")
    print("="*80)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_tabular, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_tabular = X_tabular.to(device)
            
            # No categorical features for CICIDS
            X_cat = torch.zeros(X_tabular.size(0), 0, device=device, dtype=torch.long)
            
            optimizer.zero_grad()
            loss, _ = model.forward_mfm(X_tabular, X_cat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"[PRETRAIN] Epoch {epoch+1}/{epochs} - MFM Loss: {avg_loss:.4f}")
        
        mlflow.log_metric("pretrain_loss", avg_loss, step=epoch)
    
    print("✓ Tabular Expert pretraining complete!\n")


# ============================================================
#  Train MoE End-to-End
# ============================================================
def train_moe(moe_model, dataloader, optimizer, criterion, device):
    """Train full MoE model with all experts."""
    moe_model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_gating_weights = []
    
    for X_tabular, X_temporal, y in tqdm(dataloader, desc="Training MoE"):
        X_tabular = X_tabular.to(device)
        X_temporal = X_temporal.to(device)
        y = y.to(device)
        
        # No categorical features for CICIDS
        X_cat = torch.zeros(X_tabular.size(0), 0, device=device, dtype=torch.long)
        
        optimizer.zero_grad()
        
        # Forward pass with expert outputs
        logits, z_tab, z_temp, gating_weights, z_combined = moe_model(
            X_tabular, X_cat, X_temporal, return_expert_outputs=True
        )
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        all_gating_weights.append(gating_weights.detach().cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    # Analyze gating weights
    all_gating_weights = np.concatenate(all_gating_weights, axis=0)
    metrics['mean_gating_tabular'] = all_gating_weights[:, 0].mean()
    metrics['mean_gating_temporal'] = all_gating_weights[:, 1].mean()
    metrics['std_gating_tabular'] = all_gating_weights[:, 0].std()
    metrics['std_gating_temporal'] = all_gating_weights[:, 1].std()
    
    return metrics, all_gating_weights


# ============================================================
#  Evaluate MoE
# ============================================================
def evaluate_moe(moe_model, dataloader, criterion, device):
    """Evaluate MoE with comprehensive metrics."""
    moe_model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_gating_weights = []
    
    with torch.no_grad():
        for X_tabular, X_temporal, y in dataloader:
            X_tabular = X_tabular.to(device)
            X_temporal = X_temporal.to(device)
            y = y.to(device)
            
            X_cat = torch.zeros(X_tabular.size(0), 0, device=device, dtype=torch.long)
            
            logits, z_tab, z_temp, gating_weights, z_combined = moe_model(
                X_tabular, X_cat, X_temporal, return_expert_outputs=True
            )
            
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_gating_weights.append(gating_weights.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    # Gating weight analysis
    all_gating_weights = np.concatenate(all_gating_weights, axis=0)
    metrics['mean_gating_tabular'] = all_gating_weights[:, 0].mean()
    metrics['mean_gating_temporal'] = all_gating_weights[:, 1].mean()
    
    return metrics, all_gating_weights


# ============================================================
#  Main Training Loop
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--stage", type=str, choices=["pretrain", "train"], required=True,
                       help="pretrain: Pretrain Tabular Expert only | train: Train full MoE")
    args = parser.parse_args()
    
    # Load params
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    
    dataset_name = params['data']['dataset']
    dataset_dir = f"data/processed/{dataset_name.lower()}"
    
    # Device
    device = torch.device("cuda" if params['train']['use_cuda'] and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # ============================================================
    # STAGE 1: PRETRAIN TABULAR EXPERT (on tabular features only)
    # ============================================================
    if args.stage == "pretrain":
        print("\n" + "="*80)
        print("STAGE 1: PRETRAIN TABULAR EXPERT")
        print("="*80)
        
        # Load tabular-only dataset
        train_loader, val_loader = load_tabular_only_dataset(
            dataset_dir,
            batch_size=params['train']['batch_size']
        )
        
        # Get metadata for feature count
        with open(f"{dataset_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        n_tabular = metadata['feature_split']['n_tabular']
        
        # Create Tabular Expert
        config = TabularExpertConfig(
            n_num=n_tabular,
            cat_cardinalities=[],  # No categorical for CICIDS
            d_model=params['train']['d_model'],
            n_heads=params['train']['n_heads'],
            n_layers=params['train']['n_layers'],
            clf_hidden=params['train']['clf_hidden'],
            n_classes=params['train']['n_classes'],
            dropout=params['train']['dropout'],
            mask_ratio=params['train']['mask_ratio']
        )
        
        model = TabularExpert(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['learning_rate'])
        
        # MLflow
        mlflow.set_experiment(f"MoE_{dataset_name}_Pretrain")
        with mlflow.start_run():
            mlflow.log_params(params['train'])
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("n_tabular_features", n_tabular)
            mlflow.log_param("stage", "pretrain_tabular")
            
            # Pretrain
            pretrain_tabular_expert(
                model, train_loader, optimizer, device,
                epochs=params['train']['epochs_pretrain']
            )
            
            # Save pretrained weights
            os.makedirs("models/weights", exist_ok=True)
            save_path = f"models/weights/{dataset_name.lower()}_tabular_pretrained.pt"
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
            print(f"✓ Pretrained Tabular Expert saved to: {save_path}")
    
    # ============================================================
    # STAGE 2: TRAIN FULL MOE (end-to-end)
    # ============================================================
    elif args.stage == "train":
        print("\n" + "="*80)
        print("STAGE 2: TRAIN FULL MOE MODEL")
        print("="*80)
        
        # Load MoE dataset
        train_loader, val_loader, metadata = load_moe_dataset(
            dataset_dir,
            batch_size=params['train']['batch_size']
        )
        
        n_tabular = metadata['feature_split']['n_tabular']
        n_temporal = metadata['feature_split']['n_temporal']
        
        print(f"\n[INFO] Creating MoE model:")
        print(f"       Dataset: {dataset_name}")
        print(f"       Tabular features: {n_tabular}")
        print(f"       Temporal features: {n_temporal}")
        
        # Check for pretrained Tabular Expert
        pretrained_path = f"models/weights/{dataset_name.lower()}_tabular_pretrained.pt"
        if not os.path.exists(pretrained_path):
            print(f"\n[WARNING] Pretrained Tabular Expert not found at: {pretrained_path}")
            print(f"          Run: python -m src.models.train_moe --stage pretrain")
            print(f"          Training without pretrained weights...")
            pretrained_path = None
        else:
            print(f"[INFO] Found pretrained Tabular Expert: {pretrained_path}")
        
        # Create MoE model
        moe_model = create_moe_model(
            dataset=dataset_name,
            n_tabular_num=n_tabular,
            n_tabular_cat=0,  # No categorical for CICIDS
            cat_cardinalities=None,
            d_embedding=params['train']['d_model'],
            clf_hidden=params['train']['clf_hidden'],  # Match pretraining config
            pretrained_tabular_path=pretrained_path,
            sparse_gating=False,
            freeze_tabular=False  # Train all experts end-to-end
        )
        
        moe_model = moe_model.to(device)
        
        # Calculate class weights
        y_train = torch.cat([y for _, _, y in train_loader])
        class_weights = calculate_class_weights(y_train, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        optimizer = torch.optim.Adam(moe_model.parameters(), lr=params['train']['learning_rate'])
        
        # MLflow
        mlflow.set_experiment(f"MoE_{dataset_name}")
        with mlflow.start_run():
            mlflow.log_params(params['train'])
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("n_tabular_features", n_tabular)
            mlflow.log_param("n_temporal_features", n_temporal)
            mlflow.log_param("pretrained_tabular", pretrained_path is not None)
            mlflow.log_param("stage", "train_moe")
            
            best_val_f1 = 0.0
            
            print("\n" + "="*80)
            print("TRAINING MoE MODEL")
            print("="*80)
            
            for epoch in range(params['train']['epochs_finetune']):
                print(f"\n[EPOCH {epoch+1}/{params['train']['epochs_finetune']}]")
                
                # Train
                train_metrics, train_gating = train_moe(moe_model, train_loader, optimizer, criterion, device)
                
                # Evaluate
                val_metrics, val_gating = evaluate_moe(moe_model, val_loader, criterion, device)
                
                # Log metrics
                mlflow.log_metric("train_loss", train_metrics['loss'], step=epoch)
                mlflow.log_metric("train_f1", train_metrics['f1'], step=epoch)
                mlflow.log_metric("train_precision", train_metrics['precision'], step=epoch)
                mlflow.log_metric("train_recall", train_metrics['recall'], step=epoch)
                mlflow.log_metric("train_auc_pr", train_metrics['auc_pr'], step=epoch)
                
                mlflow.log_metric("val_loss", val_metrics['loss'], step=epoch)
                mlflow.log_metric("val_f1", val_metrics['f1'], step=epoch)
                mlflow.log_metric("val_precision", val_metrics['precision'], step=epoch)
                mlflow.log_metric("val_recall", val_metrics['recall'], step=epoch)
                mlflow.log_metric("val_auc_pr", val_metrics['auc_pr'], step=epoch)
                
                # Gating weights
                mlflow.log_metric("train_gating_tabular", train_metrics['mean_gating_tabular'], step=epoch)
                mlflow.log_metric("train_gating_temporal", train_metrics['mean_gating_temporal'], step=epoch)
                mlflow.log_metric("val_gating_tabular", val_metrics['mean_gating_tabular'], step=epoch)
                mlflow.log_metric("val_gating_temporal", val_metrics['mean_gating_temporal'], step=epoch)
                
                # Print summary
                print(f"\n[TRAIN] Loss: {train_metrics['loss']:.4f} | F1: {train_metrics['f1']:.4f} | "
                      f"Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f} | "
                      f"AUC-PR: {train_metrics['auc_pr']:.4f}")
                print(f"        Gating: Tabular={train_metrics['mean_gating_tabular']:.3f}, "
                      f"Temporal={train_metrics['mean_gating_temporal']:.3f}")
                
                print(f"\n[VAL]   Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['f1']:.4f} | "
                      f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
                      f"AUC-PR: {val_metrics['auc_pr']:.4f}")
                print(f"        Gating: Tabular={val_metrics['mean_gating_tabular']:.3f}, "
                      f"Temporal={val_metrics['mean_gating_temporal']:.3f}")
                
                # Confusion matrix
                cm = val_metrics['confusion_matrix']
                print(f"\n[VAL] Confusion Matrix:")
                print(f"        [[TN={val_metrics['tn']}, FP={val_metrics['fp']}],")
                print(f"         [FN={val_metrics['fn']}, TP={val_metrics['tp']}]]")
                print(f"        Missed attacks: {val_metrics['fn']} | False alarms: {val_metrics['fp']}")
                
                # Save best model
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    save_path = f"models/weights/{dataset_name.lower()}_moe_best.pt"
                    torch.save(moe_model.state_dict(), save_path)
                    print(f"\n✓ New best F1-Score: {best_val_f1:.4f} - Model saved!")
            
            # Save final model
            final_save_path = f"models/weights/{dataset_name.lower()}_moe_final.pt"
            torch.save(moe_model.state_dict(), final_save_path)
            mlflow.log_artifact(final_save_path)
            
            print("\n" + "="*80)
            print(f"✅ MoE TRAINING COMPLETE!")
            print(f"   Best Val F1-Score: {best_val_f1:.4f}")
            print(f"   Final model saved to: {final_save_path}")
            print("="*80)


if __name__ == "__main__":
    main()
