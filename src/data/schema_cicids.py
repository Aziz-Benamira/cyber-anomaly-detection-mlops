from dataclasses import dataclass
from typing import List

@dataclass
class CICIDSSchema:
    # Pas de vraie timestamp, mais Flow Duration peut jouer ce rôle
    time_col: str = "Flow Duration"

    # CICIDS version Kaggle: aucune colonne catégorielle (tout est numérique sauf le label)
    cat_cols: List[str] = ()

    # Colonnes numériques principales (toutes sauf Label)
    num_cols: List[str] = (
        "Destination Port", "Flow Duration",
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "Fwd Header Length", "Bwd Header Length",
        "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "CWE Flag Count", "ECE Flag Count",
        "Down/Up Ratio", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "Fwd Header Length.1",
        "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
    )

    # Colonne cible (BENIGN ou attaque)
    label_col: str = "Label"

    @property
    def all_features(self) -> List[str]:
        return list(self.cat_cols) + list(self.num_cols)
