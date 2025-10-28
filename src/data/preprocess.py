from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

from src.data.dataset import load_unsw_nb15, load_cicids2017
from src.data.schema_unsw import UNSWSchema
from src.data.schema_cicids import CICIDSSchema


def load_dataset(dataset_name: str, raw_dir: str) -> pd.DataFrame:
    """Charger le dataset brut"""
    if dataset_name.upper() == "UNSW":
        return load_unsw_nb15(raw_dir), UNSWSchema()
    elif dataset_name.upper() == "CICIDS":
        return load_cicids2017(raw_dir), CICIDSSchema()
    else:
        raise ValueError(f"Dataset inconnu: {dataset_name}")


def build_preprocessor(cat_cols, num_cols):
    """Pipeline sklearn pour transformer cat + num"""
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])


def preprocess(df: pd.DataFrame, schema, out_dir: str, max_samples: int = None, balance_classes: bool = True):
    """
    Encoder features et séparer X/y
    
    Args:
        df: Input dataframe
        schema: Dataset schema (UNSW or CICIDS)
        out_dir: Output directory
        max_samples: Maximum number of samples to keep (None = all)
        balance_classes: If True and max_samples is set, balance attack/normal classes
    """

    # Nettoyage colonnes (strip pour CICIDS qui a des espaces)
    df.columns = df.columns.str.strip()

    # Extraire X et y
    X = df[schema.all_features]
    y = df[schema.label_col] if schema.label_col in df.columns else None

    # ===== LABEL ENCODING =====
    # Convert string labels to binary integers: 0 = BENIGN/Normal, 1 = Attack
    if y is not None:
        y_original = y.copy()
        if y.dtype == object or pd.api.types.is_string_dtype(y):
            # Map: BENIGN/Normal -> 0, anything else -> 1
            y = y.astype(str).str.strip().str.lower()
            y = y.apply(lambda x: 0 if x in ['benign', 'normal'] else 1)
            y = y.astype(np.int64)
            
            print(f"[INFO] Label encoding applied:")
            print(f"       Original labels: {y_original.unique()[:10]}")
            print(f"       Encoded distribution: {np.bincount(y)}")
            print(f"       Class 0 (Normal): {(y==0).sum()}, Class 1 (Attack): {(y==1).sum()}")
        else:
            y = y.astype(np.int64)
    
    # ===== BALANCED SAMPLING (for mini datasets) =====
    if max_samples is not None and y is not None and balance_classes:
        # Get indices for each class
        normal_idx = np.where(y == 0)[0]
        attack_idx = np.where(y == 1)[0]
        
        samples_per_class = max_samples // 2
        
        # Sample from each class
        if len(normal_idx) > samples_per_class:
            normal_idx = np.random.choice(normal_idx, samples_per_class, replace=False)
        if len(attack_idx) > samples_per_class:
            attack_idx = np.random.choice(attack_idx, samples_per_class, replace=False)
        
        # Combine and shuffle
        selected_idx = np.concatenate([normal_idx, attack_idx])
        np.random.shuffle(selected_idx)
        
        X = X.iloc[selected_idx]
        y = y.iloc[selected_idx] if hasattr(y, 'iloc') else y[selected_idx]
        
        print(f"[INFO] Balanced sampling: {len(selected_idx)} samples")
        print(f"       Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

    # ===== HANDLE MISSING VALUES (before encoding) =====
    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN: numeric columns with median, categorical with mode
    for col in schema.num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')  # Force numeric
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
    
    for col in schema.cat_cols:
        if col in X.columns:
            if X[col].isna().any():
                mode_val = X[col].mode()[0] if not X[col].mode().empty else 'unknown'
                X[col] = X[col].fillna(mode_val) 
    
    # Encoder
    preproc = build_preprocessor(schema.cat_cols, schema.num_cols)
    X_trans = preproc.fit_transform(X)

    # Sauvegarder objets
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dump(preproc, f"{out_dir}/preprocessor.joblib")
    
    # Save metadata about the preprocessing
    metadata = {
        "n_samples": len(X_trans),
        "n_features_original": len(schema.all_features),
        "n_features_after_encoding": X_trans.shape[1],
        "n_categorical_original": len(schema.cat_cols),
        "n_numeric_original": len(schema.num_cols),
        "categorical_columns": list(schema.cat_cols),
        "numeric_columns": list(schema.num_cols),
        "label_encoding": {"0": "Normal/Benign", "1": "Attack"}
    }
    
    import json
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Sauver en numpy (make sure y is numpy array)
    np.save(f"{out_dir}/X.npy", X_trans)
    if y is not None:
        y_array = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)
        np.save(f"{out_dir}/y.npy", y_array)

    print(f"[INFO] Saved processed data to {out_dir}")
    print(f"       X shape = {X_trans.shape}, y shape = {None if y is None else y_array.shape}")
    print(f"       Features: {len(schema.num_cols)} numeric + {len(schema.cat_cols)} categorical")
    print(f"       After encoding: {X_trans.shape[1]} features (OneHot expanded)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--mini", action="store_true", help="Create mini dataset for faster experimentation")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples for mini dataset")
    args = parser.parse_args()

    # Charger config
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)

    dataset_name = params["data"]["dataset"]
    raw_dir = "data/raw"

    # le sous-dossier dépend du dataset (unsw/ ou cicids/)
    out_dir = Path(args.out) / dataset_name.lower()

    df, schema = load_dataset(dataset_name, raw_dir)

    # Downsample options
    max_rows = params["preprocess"].get("max_rows", None)
    if max_rows in [None, "None", "null"]:
        max_rows = None
    else:
        max_rows = int(max_rows)

    # Use --mini flag for balanced mini dataset
    if args.mini:
        max_rows = args.max_samples
        print(f"[INFO] Creating MINI dataset with max {max_rows} balanced samples")
        balance = True
    else:
        balance = False

    if max_rows:
        # Don't use head() - let preprocess() do balanced sampling
        print(f"[INFO] Will sample {max_rows} rows with balance={balance}")

    preprocess(df, schema, out_dir, max_samples=max_rows, balance_classes=balance)