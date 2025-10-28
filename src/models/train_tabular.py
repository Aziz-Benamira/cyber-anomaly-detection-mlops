import os
import yaml
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.models.tabular_expert import TabularExpert, TabularExpertConfig


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
    model.train()
    total_loss, correct, total = 0, 0, 0

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
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X_num = X
            X_cat = torch.zeros(X.size(0), 0, device=device, dtype=torch.long)

            logits = model.forward_classify(X_num, X_cat)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(dataloader), correct / total


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
    criterion = nn.CrossEntropyLoss()

    # Start MLflow experiment
    experiment_name = f"tabular_expert_{dataset}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{stage}_{dataset}"):

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
            
            for epoch in range(epochs):
                train_loss, train_acc = train_supervised(model, train_dl, optimizer, criterion, device)
                val_loss, val_acc = evaluate(model, val_dl, criterion, device)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

                print(f"[Epoch {epoch+1}/{epochs}] Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f}")

            Path("models/weights").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), finetuned_path)
            mlflow.log_artifact(finetuned_path)
            
            print(f"\n✅ Finetune complete! Saved to: {finetuned_path}")
            print(f"   Final performance: Train={train_acc:.3f} | Val={val_acc:.3f}")

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
