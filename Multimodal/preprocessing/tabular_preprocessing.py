"""
Tabular (Metadata) Preprocessing — ISIC 2024
Dataset: train-metadata.csv — 400k hang, ~50 cot

Thay doi so voi ISIC 2019:
  - ID anh  : isic_id  (cu: image_name)
  - Nhan    : target 0/1 (cu: diagnosis string 7 lop)
  - Site col: anatom_site_general (cu: anatom_site_general_challenge)
  - Them 40+ cot tbp_lv_* (dac trung hinh hoc TBP)
  - Cross-val: StratifiedGroupKFold theo patient_id (tranh data leakage)
  - Feature dim: 9 chieu (cu: 5 chieu)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dinh nghia cot
# ─────────────────────────────────────────────────────────────────────────────

# ISIC 2024: binary
CLASS_NAMES = ["benign", "malignant"]
TARGET_COL  = "target"
ID_COL      = "isic_id"
GROUP_COL   = "patient_id"    # dung cho StratifiedGroupKFold

# Cot metadata co ban (gion voi ISIC 2019)
SITE_COL = "anatom_site_general"   # ISIC 2024 doi ten (bo _challenge)

SITE_CATEGORIES = [
    "head/neck", "upper extremity", "lower extremity",
    "torso", "palms/soles", "oral/genital", "unknown",
]
SEX_CATEGORIES = ["male", "female", "unknown"]

# Cot dac trung hinh hoc TBP (tbp_lv_*) — 40+ cot trong ISIC 2024
# Day la tap con quan trong nhat theo EDA community
TBP_FEATURE_COLS = [
    "tbp_lv_A",           # chanel A trong khong gian mau LAB
    "tbp_lv_Aext",        # A mo rong
    "tbp_lv_B",           # chanel B trong LAB
    "tbp_lv_Bext",
    "tbp_lv_C",           # Chroma (do bao hoa)
    "tbp_lv_Cext",
    "tbp_lv_H",           # Hue angle
    "tbp_lv_Hext",
    "tbp_lv_L",           # Luminance
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",     # Dien tich ton thuong (mm2)
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",      # Su khac biet mau giua ton thuong va da xung quanh
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLB",
    "tbp_lv_eccentricity",  # Do lech tam (0=tron, 1=duong thang)
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",  # Do tin cay day la nevi (0-1)
    "tbp_lv_norm_border",    # Vien bien chuan hoa
    "tbp_lv_norm_color",     # Mau sac chuan hoa
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_mean",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",    # Doi xung 2 truc
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_x",             # Vi tri X tren co the
    "tbp_lv_y",
    "tbp_lv_z",
]

# Cac feature cuoi sau engineer_features: co ban + TBP + derived
# Tong: 9 chieu (3 co ban + 5 derived + 1 tbp_nevi_confidence giu lai)
BASE_FEATURE_COLS = [
    "age_approx",
    "is_male",
    "high_risk_site",
    "age_bucket",
    "site_encoded",
    "tbp_lv_nevi_confidence",  # feature TBP quan trong nhat (1 cot dai dien)
    "tbp_color_asymmetry",     # derived tu tbp_lv_*
    "tbp_size_mm2_norm",       # derived
    "tbp_border_color_score",  # derived
]
FEATURE_DIM = len(BASE_FEATURE_COLS)  # 9


# ─────────────────────────────────────────────────────────────────────────────
# Lam sach du lieu
# ─────────────────────────────────────────────────────────────────────────────
def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuan hoa raw CSV cua ISIC 2024:
      - Xoa hang khong co isic_id
      - Fill missing: age dung median, sex/site dung 'unknown'
      - Clip age [0, 110]
      - Xu ly cot target (0/1)
      - Chuyen ten cot site ve dinh dang chuan
    """
    df = df.copy()

    # Xoa hang thieu ID
    df = df.dropna(subset=[ID_COL])

    # Chuan hoa ten cot site (ISIC 2024 doi ten so voi 2019)
    if SITE_COL not in df.columns and "anatom_site_general_challenge" in df.columns:
        df = df.rename(columns={"anatom_site_general_challenge": SITE_COL})

    # Chuan hoa chuoi
    for col in ["sex", SITE_COL]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("unknown")
                .astype(str)
                .str.lower()
                .str.strip()
            )

    # Tuoi
    if "age_approx" in df.columns:
        median_age = df["age_approx"].median()
        if pd.isna(median_age):
            median_age = 50.0
        df["age_approx"] = (
            df["age_approx"].fillna(median_age).clip(lower=0, upper=110)
        )

    # Nhan binary
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    # Cac cot TBP: fill missing bang median
    tbp_present = [c for c in TBP_FEATURE_COLS if c in df.columns]
    for col in tbp_present:
        med = df[col].median()
        df[col] = df[col].fillna(med if not pd.isna(med) else 0.0)

    log.info("clean_metadata: %d hang sau khi lam sach", len(df))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tao cac dac trung phu tu metadata ISIC 2024.
    Ket hop ca metadata co ban va dac trung TBP derived.
    """
    df = df.copy()

    # --- Metadata co ban (gion voi ISIC 2019) --------------------------------
    df["age_bucket"] = pd.cut(
        df["age_approx"],
        bins=[0, 20, 40, 60, 80, 110],
        labels=[0, 1, 2, 3, 4],
    ).astype(float).fillna(2.0)

    high_risk_sites = {"torso", "head/neck"}
    if SITE_COL in df.columns:
        df["high_risk_site"] = df[SITE_COL].isin(high_risk_sites).astype(float)
    else:
        df["high_risk_site"] = 0.0

    if "sex" in df.columns:
        df["is_male"] = (df["sex"] == "male").astype(float)
    else:
        df["is_male"] = 0.5

    # Encode site thanh so
    site_map = {s: i for i, s in enumerate(SITE_CATEGORIES)}
    if SITE_COL in df.columns:
        df["site_encoded"] = (
            df[SITE_COL]
            .map(site_map)
            .fillna(len(SITE_CATEGORIES) - 1)  # unknown
            .astype(float)
        )
    else:
        df["site_encoded"] = float(len(SITE_CATEGORIES) - 1)

    # --- Dac trung TBP derived -----------------------------------------------
    # Color asymmetry: su bat doi xung mau giua kenh A va B
    if "tbp_lv_deltaA" in df.columns and "tbp_lv_deltaB" in df.columns:
        df["tbp_color_asymmetry"] = np.sqrt(
            df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2
        ).fillna(0.0)
    else:
        df["tbp_color_asymmetry"] = 0.0

    # Kich thuoc chuan hoa (log scale)
    if "tbp_lv_areaMM2" in df.columns:
        df["tbp_size_mm2_norm"] = np.log1p(
            df["tbp_lv_areaMM2"].clip(lower=0)
        ).fillna(0.0)
    else:
        df["tbp_size_mm2_norm"] = 0.0

    # Diem bien & mau tong hop
    if "tbp_lv_norm_border" in df.columns and "tbp_lv_norm_color" in df.columns:
        df["tbp_border_color_score"] = (
            df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"]
        ).fillna(0.0)
    else:
        df["tbp_border_color_score"] = 0.0

    # Nevi confidence (giu nguyen neu co)
    if "tbp_lv_nevi_confidence" not in df.columns:
        df["tbp_lv_nevi_confidence"] = 0.5

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MetadataPreprocessor — fit tren train, transform tren tat ca splits
# ─────────────────────────────────────────────────────────────────────────────
class MetadataPreprocessor:
    """
    Stateful preprocessor cho ISIC 2024 metadata.
    Output: vector 9 chieu float32 cho MetadataMLP.

    Feature vector:
      [age_approx, is_male, high_risk_site, age_bucket, site_encoded,
       tbp_lv_nevi_confidence, tbp_color_asymmetry,
       tbp_size_mm2_norm, tbp_border_color_score]
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM  # 9

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        df_eng = engineer_features(df)
        X = self._extract(df_eng)
        self.scaler.fit(X)
        self._fitted = True
        log.info("MetadataPreprocessor fitted: %d samples, %d features", len(df), FEATURE_DIM)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor chua duoc fit. Goi .fit() truoc.")
        df_eng = engineer_features(df)
        X = self._extract(df_eng)
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(self.transform(df), dtype=torch.float32)

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        """Lay cac cot feature, fill 0 neu cot chua ton tai."""
        out = pd.DataFrame(index=df.index)
        for col in BASE_FEATURE_COLS:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                out[col] = 0.0
        return out.values.astype(np.float64)

    def attach_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Them cot 'meta_features' (list float) vao DataFrame.
        ISICDataset doc cot nay trong __getitem__.
        """
        features = self.transform(df)
        df = df.copy()
        df["meta_features"] = [features[i].tolist() for i in range(len(features))]
        return df

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "feature_dim": FEATURE_DIM}, path)
        log.info("Preprocessor saved: %s", path)

    @classmethod
    def load(cls, path: str) -> "MetadataPreprocessor":
        obj = cls()
        state = joblib.load(path)
        obj.scaler = state["scaler"]
        obj._fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation — PHAI dung GroupKFold theo patient_id
# ─────────────────────────────────────────────────────────────────────────────
def create_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tao cot 'fold' bang StratifiedGroupKFold.

    QUAN TRONG: ISIC 2024 co nhieu anh cung mot benh nhan (patient_id).
    Neu dung StratifiedKFold thong thuong, anh cung benh nhan co the
    o ca train va val -> DATA LEAKAGE -> AUC ao cao.

    StratifiedGroupKFold dam bao:
      - Moi benh nhan chi xuat hien o dung 1 fold (train hoac val)
      - Phan phoi nhan binary duoc bao toan giua cac fold
    """
    df = df.copy()
    df["fold"] = -1

    if GROUP_COL not in df.columns:
        log.warning(
            "Khong tim thay cot '%s' — su dung StratifiedKFold thay the "
            "(co nguy co data leakage neu dataset co nhieu anh/benh nhan).",
            GROUP_COL,
        )
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(skf.split(df, df[TARGET_COL])):
            df.loc[val_idx, "fold"] = fold_idx
        return df

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(df, df[TARGET_COL], groups=df[GROUP_COL])
    ):
        df.loc[val_idx, "fold"] = fold_idx

    log.info(
        "create_folds: %d fold | %d benh nhan | target: %d pos / %d neg",
        n_splits,
        df[GROUP_COL].nunique(),
        (df[TARGET_COL] == 1).sum(),
        (df[TARGET_COL] == 0).sum(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Tinh trong so lop cho BCEWithLogitsLoss
# ─────────────────────────────────────────────────────────────────────────────
def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    """
    pos_weight = n_negative / n_positive cho BCEWithLogitsLoss.
    ISIC 2024: ~33 (malignant ~3%) -> loss nhan manh vao duong tinh.
    """
    n_pos = (df[TARGET_COL] == 1).sum()
    n_neg = (df[TARGET_COL] == 0).sum()
    if n_pos == 0:
        return torch.tensor(1.0)
    pw = n_neg / n_pos
    log.info("pos_weight = %.2f (n_pos=%d, n_neg=%d)", pw, n_pos, n_neg)
    return torch.tensor(pw, dtype=torch.float32)


# Alias de giu tuong thich voi cac module khac
def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Alias cua compute_pos_weight cho binary task."""
    return compute_pos_weight(df)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Tabular Preprocessing Module (ISIC 2024) ===")
    rng = np.random.default_rng(42)
    n = 200
    data = {
        "isic_id":   [f"ISIC_{i:07d}" for i in range(n)],
        "patient_id":[f"PAT_{i//4:04d}" for i in range(n)],
        "target":     rng.choice([0, 1], n, p=[0.97, 0.03]),
        "age_approx": rng.choice([25., 40., 55., 70., np.nan], n),
        "sex":        rng.choice(["male", "female", None], n),
        SITE_COL:     rng.choice(SITE_CATEGORIES + [None], n),
        "tbp_lv_nevi_confidence": rng.uniform(0, 1, n),
        "tbp_lv_deltaA": rng.normal(0, 5, n),
        "tbp_lv_deltaB": rng.normal(0, 5, n),
        "tbp_lv_areaMM2": rng.uniform(1, 50, n),
        "tbp_lv_norm_border": rng.uniform(0, 1, n),
        "tbp_lv_norm_color":  rng.uniform(0, 1, n),
    }
    df_raw = pd.DataFrame(data)
    df = clean_metadata(df_raw)
    df = create_folds(df)
    print(f"Shape: {df.shape} | Folds: {sorted(df['fold'].unique())}")

    train_df = df[df["fold"] != 0]
    val_df   = df[df["fold"] == 0]
    pp = MetadataPreprocessor()
    X_train = pp.fit_transform(train_df)
    X_val   = pp.transform(val_df)
    print(f"Train features: {X_train.shape} | Val features: {X_val.shape}")
    print(f"Feature dim   : {pp.feature_dim}")
    print(f"pos_weight    : {compute_pos_weight(df).item():.2f}")
    print("OK")
