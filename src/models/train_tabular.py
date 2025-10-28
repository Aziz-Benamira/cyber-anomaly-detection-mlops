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

from src.models.tabular_expert import TabularExpert, TabularExpertConfig


# ============================================================
#  Class Weight Calculation for Imbalanced Data
# ============================================================
def calculate_class_weights(y_tensor, device):
    """
    Calculate class weights for imbalanced datasets.
    
    Formula: weight_i = total_samples / (num_classes * count_i)
    This heavily penalizes errors on minority class.
    
    Args:
        y_tensor: Label tensor (torch.Tensor)
        device: Device to put weights on
        
    Returns:
        torch.Tensor: Class weights [weight_class_0, weight_class_1]
    """
    # Get class counts
    unique, counts = torch.unique(y_tensor, return_counts=True)
    total_samples = len(y_tensor)
    num_classes = len(unique)
    
    # Calculate weights
    weights = torch.zeros(num_classes, device=device)
    for class_idx, count in zip(unique, counts):
        weights[class_idx] = total_samples / (num_classes * count)
    
    print(f"\n[INFO] Class Weight Calculation:")
    print(f"       Total samples: {total_samples:,}")
    print(f"       Class distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"         Class {class_idx}: {count:,} samples ({count/total_samples*100:.1f}%) -> weight={weights[class_idx]:.4f}")
    print(f"       Weight ratio (attack/normal): {weights[1]/weights[0]:.2f}x")
    print(f"       ⚠️  Minority class errors will be penalized {weights[1]/weights[0]:.2f}x more!\n")
    
    return weights


# ============================================================
#  Load dataset from numpy
# ============================================================
def load_dataset(dataset_dir, batch_size=256, test_size=0.2, random_state=42):
    """
    Loads X.npy and y.npy from the given directory.
    Uses STRATIFIED split to ensure both classes in train/val sets.
    
    Returns:
        train_loader, val_loader
    """
    # Load arrays
    X = np.load(f"{dataset_dir}/X.npy", allow_pickle=True)
    y = np.load(f"{dataset_dir}/y.npy", allow_pickle=True)

    print(f"[INFO] Loaded dataset from {dataset_dir}")
    print(f"       X shape: {X.shape}, dtype: {X.dtype}")
    print(f"       y shape: {y.shape}, dtype: {y.dtype}")

    # --- Fix dtype issues ---
    # If X is not numeric, convert safely
    if not np.issubdtype(X.dtype, np.number):
        X = X.astype(np.float32)

    # Handle labels - MUST be integers!
    if y.dtype == object or np.issubdtype(y.dtype, np.str_):
        print("[ERROR] Labels are still strings! Preprocessing failed.")
        print(f"        Sample labels: {y[:10]}")
        raise ValueError("Labels must be encoded as integers. Re-run preprocessing with fixed script.")
    
    y = y.astype(np.int64)
    
    # Check label distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"[INFO] Label distribution: {dict(zip(unique, counts))}")
    
    if len(unique) < 2:
        print("[WARNING] Only one class in dataset! Model will trivially get 100% accuracy.")
        print("          Re-run preprocessing with balanced sampling!")
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # STRATIFIED split to ensure both classes in train/val
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y.numpy()  # KEY: stratified by labels
    )
    
    print(f"[INFO] Stratified split: {len(train_idx)} train, {len(val_idx)} val")
    print(f"       Train labels: {np.bincount(y[train_idx].numpy())}")
    print(f"       Val labels:   {np.bincount(y[val_idx].numpy())}")

    # Create datasets
    ds = TensorDataset(X, y)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ============================================================
#  Training functions
# ============================================================

def train_self_supervised(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for X, _ in tqdm(dataloader, desc="Pretraining"):
        X = X.to(device)

        # Split numeric vs categorical if needed
        X_num = X  # assume all numeric for now
        X_cat = torch.zeros(X.size(0), 0, device=device, dtype=torch.long)

        optimizer.zero_grad()
        loss, _ = model.forward_mfm(X_num, X_cat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_supervised(model, dataloader, optimizer, criterion, device):
    """Train with supervised classification loss"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for X, y in tqdm(dataloader, desc="Fine-tuning"):
        X, y = X.to(device), y.to(device)
        X_num = X
        X_cat = torch.zeros(X.size(0), 0, device=device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model.forward_classify(X_num, X_cat)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Collect predictions for metrics
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())  # Probability of attack class

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate with comprehensive metrics (not just accuracy!)"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X_num = X
            X_cat = torch.zeros(X.size(0), 0, device=device, dtype=torch.long)

            logits = model.forward_classify(X_num, X_cat)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            # Collect predictions
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate comprehensive metrics for imbalanced anomaly detection.
    
    Focus on ATTACK class (class 1) metrics:
    - Precision: Of predicted attacks, how many are real?
    - Recall: Of real attacks, how many did we catch?
    - F1-Score: Harmonic mean of precision and recall
    - AUC-PR: Area under precision-recall curve
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for attack class
        
    Returns:
        dict: Comprehensive metrics
    """
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
    
    # AUC-PR (more informative than ROC-AUC for imbalanced data)
    try:
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        auc_pr = auc(recalls, precisions)
    except:
        auc_pr = 0.0
    
    # ROC-AUC (for reference)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.0
    
    # Overall accuracy (less important for imbalanced data!)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,  # Keep for comparison, but don't rely on it!
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_pr': auc_pr,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


# ============================================================
#  Main training entrypoint
# ============================================================

def main(params_path="params.yaml", stage="pretrain"):
    import numpy as np

    # Load config
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Auto-detect dataset from params
    dataset = params["data"]["dataset"].lower()
    dataset_dir = f"data/processed/{dataset}"
    
    # Auto-generate paths based on dataset
    pretrained_path = f"models/weights/{dataset}_pretrained.pt"
    finetuned_path = f"models/weights/{dataset}_finetuned.pt"
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Stage: {stage.upper()}")
    print(f"Data directory: {dataset_dir}")
    print(f"{'='*60}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and params["train"].get("use_cuda", True) else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create Dataloaders with stratified split
    train_dl, val_dl = load_dataset(
        dataset_dir, 
        batch_size=params["train"]["batch_size"],
        test_size=0.2,
        random_state=42
    )

    # Load metadata to get feature counts
    import joblib
    preprocessor_path = f"{dataset_dir}/preprocessor.joblib"
    
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        
        # Get the actual number of features after transformation
        # The preprocessor outputs all features (onehot-encoded cats + scaled nums)
        sample_X = train_dl.dataset.dataset.tensors[0][:1]  # Get one sample
        n_features = sample_X.shape[1]
        
        print(f"[INFO] Detected {n_features} features after preprocessing")
        
        # For TabTransformer: all features are treated as numeric after OneHotEncoding
        n_num = n_features
        cat_cardinalities = []
    else:
        # Fallback if no preprocessor found
        sample_X = train_dl.dataset.dataset.tensors[0][:1]
        n_num = sample_X.shape[1]
        cat_cardinalities = []
        print(f"[WARNING] No preprocessor found, using default feature count: {n_num}")

    # Build config for model from params
    cfg = TabularExpertConfig(
        n_num=n_num,
        cat_cardinalities=cat_cardinalities,  # Empty since OneHot already applied
        d_model=params["train"].get("d_model", 64),
        n_heads=params["train"].get("n_heads", 4),
        n_layers=params["train"].get("n_layers", 2),
        clf_hidden=params["train"].get("clf_hidden", 128),
        n_classes=params["train"]["n_classes"],
        dropout=params["train"].get("dropout", 0.1),
        mask_ratio=params["train"].get("mask_ratio", 0.15)
    )

    model = TabularExpert(cfg).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["train"]["learning_rate"])
    
    # ===== WEIGHTED LOSS for IMBALANCED DATA =====
    # Calculate class weights from training data
    train_labels = train_dl.dataset.dataset.tensors[1][train_dl.dataset.indices]
    class_weights = calculate_class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[INFO] Using WEIGHTED CrossEntropyLoss for imbalanced data")
    print(f"       Class weights: {class_weights.cpu().numpy()}")

    # Start MLflow experiment
    experiment_name = f"tabular_expert_{dataset}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{stage}_{dataset}"):

        # Log model config and training params
        mlflow.log_params({
            "dataset": dataset,
            "stage": stage,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "n_features": cfg.n_num,
            "batch_size": params["train"]["batch_size"],
            "lr": params["train"]["learning_rate"],
            "mask_ratio": cfg.mask_ratio,
            "weighted_loss": True,
            "class_weight_0": float(class_weights[0]),
            "class_weight_1": float(class_weights[1]),
        })

        if stage == "pretrain":
            epochs = params["train"]["epochs_pretrain"]
            print(f"\n[INFO] Starting PRETRAIN stage for {epochs} epoch(s)...")
            
            loss = train_self_supervised(model, train_dl, optimizer, device)
            mlflow.log_metric("pretrain_loss", loss)

            # Save pre-trained weights
            Path("models/weights").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), pretrained_path)
            mlflow.log_artifact(pretrained_path)
            
            print(f"\n✅ Pretrain complete! Saved to: {pretrained_path}")

        elif stage == "finetune":
            # Load pretrained weights if available
            if os.path.exists(pretrained_path):
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
                print(f"[INFO] Loaded pretrained weights from {pretrained_path}")
            else:
                print(f"[WARNING] No pretrained weights found at {pretrained_path}")
                print(f"          Training from scratch...")

            epochs = params["train"]["epochs_finetune"]
            print(f"\n[INFO] Starting FINETUNE stage for {epochs} epoch(s)...")
            print(f"       Metrics: F1-Score, Precision, Recall, AUC-PR (attack class)")
            print(f"       {'='*60}\n")
            
            best_f1 = 0.0
            
            for epoch in range(epochs):
                train_metrics = train_supervised(model, train_dl, optimizer, criterion, device)
                val_metrics = evaluate(model, val_dl, criterion, device)

                # Log ALL metrics to MLflow
                mlflow.log_metrics({
                    "train_loss": train_metrics['loss'],
                    "train_f1": train_metrics['f1_score'],
                    "train_precision": train_metrics['precision'],
                    "train_recall": train_metrics['recall'],
                    "train_auc_pr": train_metrics['auc_pr'],
                    "train_accuracy": train_metrics['accuracy'],  # For reference
                    "val_loss": val_metrics['loss'],
                    "val_f1": val_metrics['f1_score'],
                    "val_precision": val_metrics['precision'],
                    "val_recall": val_metrics['recall'],
                    "val_auc_pr": val_metrics['auc_pr'],
                    "val_accuracy": val_metrics['accuracy'],  # For reference
                }, step=epoch)

                # Console output: FOCUS ON F1, PRECISION, RECALL
                print(f"[Epoch {epoch+1}/{epochs}]")
                print(f"  Train -> F1={train_metrics['f1_score']:.3f} | P={train_metrics['precision']:.3f} | R={train_metrics['recall']:.3f} | AUC-PR={train_metrics['auc_pr']:.3f}")
                print(f"  Val   -> F1={val_metrics['f1_score']:.3f} | P={val_metrics['precision']:.3f} | R={val_metrics['recall']:.3f} | AUC-PR={val_metrics['auc_pr']:.3f}")
                print(f"  Val Confusion Matrix:")
                print(f"    {val_metrics['confusion_matrix']}")
                print(f"    TN={val_metrics['tn']}, FP={val_metrics['fp']}, FN={val_metrics['fn']}, TP={val_metrics['tp']}")
                print()
                
                # Track best F1 score
                if val_metrics['f1_score'] > best_f1:
                    best_f1 = val_metrics['f1_score']

            # Save final model
            Path("models/weights").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), finetuned_path)
            mlflow.log_artifact(finetuned_path)
            
            # Log final confusion matrix as text
            mlflow.log_text(str(val_metrics['confusion_matrix']), "confusion_matrix.txt")
            
            print(f"\n✅ Finetune complete! Saved to: {finetuned_path}")
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS (Validation Set - Attack Class Metrics):")
            print(f"{'='*60}")
            print(f"  F1-Score:    {val_metrics['f1_score']:.3f} (BEST: {best_f1:.3f})")
            print(f"  Precision:   {val_metrics['precision']:.3f}")
            print(f"  Recall:      {val_metrics['recall']:.3f}")
            print(f"  AUC-PR:      {val_metrics['auc_pr']:.3f}")
            print(f"  ROC-AUC:     {val_metrics['roc_auc']:.3f}")
            print(f"  Accuracy:    {val_metrics['accuracy']:.3f} (less important for imbalanced data)")
            print(f"\n  Confusion Matrix:")
            print(f"    {val_metrics['confusion_matrix']}")
            print(f"    True Positives  (TP): {val_metrics['tp']:,} (attacks caught)")
            print(f"    False Negatives (FN): {val_metrics['fn']:,} (attacks missed)")
            print(f"    False Positives (FP): {val_metrics['fp']:,} (false alarms)")
            print(f"    True Negatives  (TN): {val_metrics['tn']:,} (correctly identified normal)")
            print(f"{'='*60}\n")

        else:
            raise ValueError("Stage must be 'pretrain' or 'finetune'")


# ============================================================
# 4 Entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--stage", type=str, default="pretrain", help="pretrain or finetune")
    args = parser.parse_args()

    main(params_path=args.params, stage=args.stage)
