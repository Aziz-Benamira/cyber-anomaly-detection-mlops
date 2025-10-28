from dataclasses import dataclass
from typing import List

@dataclass
class UNSWSchema:
    # Pas de vraie timestamp, on utilisera 'dur' comme feature num
    time_col: str = None  

    # Colonnes catégorielles (REMOVED attack_cat - it's data leakage!)
    cat_cols: List[str] = (
        "proto", "service", "state"
    )

    # Colonnes numériques (toutes sauf id et label)
    num_cols: List[str] = (
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin",
        "tcprtt", "synack", "ackdat",
        "smean", "dmean",
        "trans_depth", "response_body_len",
        "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
        "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
    )

    # Label (0 = normal, 1 = attaque)
    label_col: str = "label"

    @property
    def all_features(self) -> List[str]:
        return list(self.cat_cols) + list(self.num_cols)
