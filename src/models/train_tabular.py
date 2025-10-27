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

    dataset = params["data"]["dataset"].lower()
    dataset_dir = f"data/processed/{dataset}"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Dataloaders
    train_dl, val_dl = load_dataset(dataset_dir, batch_size=256)

    # Build config for model
    cfg = TabularExpertConfig(
        n_num=train_dl.dataset[0][0].numel(),
        cat_cardinalities=[],
        d_model=64,
        n_heads=4,
        n_layers=2,
        clf_hidden=128,
        n_classes=2,
        dropout=0.1
    )

    model = TabularExpert(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Start MLflow experiment
    mlflow.set_experiment("cyber_anomaly_detection")

    with mlflow.start_run(run_name=f"{stage}_{dataset}"):

        mlflow.log_params({
            "dataset": dataset,
            "stage": stage,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "lr": 1e-4,
        })

        if stage == "pretrain":
            loss = train_self_supervised(model, train_dl, optimizer, device)
            mlflow.log_metric("pretrain_loss", loss)

            # Save pre-trained weights
            Path("models/weights").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"models/weights/{dataset}_pretrained.pt")
            mlflow.log_artifact(f"models/weights/{dataset}_pretrained.pt")

        elif stage == "finetune":
            # Load pretrained weights if available
            pretrained_path = f"models/weights/{dataset}_pretrained.pt"
            if os.path.exists(pretrained_path):
                model.load_state_dict(torch.load(pretrained_path))
                print(f"Loaded pretrained weights from {pretrained_path}")

            for epoch in range(5):
                train_loss, train_acc = train_supervised(model, train_dl, optimizer, criterion, device)
                val_loss, val_acc = evaluate(model, val_dl, criterion, device)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

                print(f"[Epoch {epoch+1}] Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f}")

            Path("models/weights").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"models/weights/{dataset}_finetuned.pt")
            mlflow.log_artifact(f"models/weights/{dataset}_finetuned.pt")

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
