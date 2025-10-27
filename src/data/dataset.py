from pathlib import Path
import pandas as pd


def load_unsw_nb15(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Charger le dataset UNSW-NB15 en concaténant tous les fichiers CSV présents.
    """
    files = list(Path(raw_dir).glob("UNSW_NB15*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier UNSW-NB15 trouvé dans {raw_dir}")
    dfs = [pd.read_csv(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] UNSW-NB15 chargé : {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def load_cicids2017(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Charger le dataset CICIDS2017 en concaténant tous les fichiers CSV présents.
    """
    files = list(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier CICIDS2017 trouvé dans {raw_dir}")
    dfs = [pd.read_csv(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] CICIDS2017 chargé : {len(df)} lignes, {len(df.columns)} colonnes")
    return df
