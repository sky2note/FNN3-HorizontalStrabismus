from __future__ import annotations
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

# --------------------------- #
# helper: winsorize by IQR
# --------------------------- #
def _winsorize_series(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return s.clip(lower, upper)

# --------------------------- #
class StrabismusDataset(Dataset):
    """
    Horizontal-strabismus multi-label *classification* dataset
    with derived features + robust preprocessing.
    """
    def __init__(
        self,
        csv_path: str | Path,
        feature_cols: List[str],
        label_cols: List[str],
        is_train: bool = True,
        scaler: Optional[StandardScaler] = None,
        dtype: torch.dtype = torch.float32,
        iqr_k: float = 1.5,
        noise_scale: float = 0.01,  # σ multiplier for noisy impute
    ) -> None:
        super().__init__()
        self.csv_path   = Path(csv_path).expanduser()
        self.feature_in = feature_cols         # 入参（含原 + 派生）
        self.label_cols = label_cols
        self.is_train   = is_train
        self.dtype      = dtype

        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        # 1 ── read csv
        df = pd.read_csv(self.csv_path)

        # 2 ── create derived columns (idempotent)
        if {"AxialLengthOD", "AxialLengthOS"}.issubset(df.columns):
            df["AL_diff"] = df["AxialLengthOD"] - df["AxialLengthOS"]
        if {"SphericalEquivalentOD", "SphericalEquivalentOS"}.issubset(df.columns):
            df["SE_diff"]  = df["SphericalEquivalentOD"] - df["SphericalEquivalentOS"]
            se_abs_sum     = df["SphericalEquivalentOD"].abs() + df["SphericalEquivalentOS"].abs() + 1e-3
            df["SE_ratio"] = df["SE_diff"] / se_abs_sum
        # age bucket & cyclical
        if "Age" in df.columns:
            bins  = [0, 5, 12, 18, 40, np.inf]
            labels = list(range(len(bins)-1))
            df["Age_bin"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False).astype(float)
            df["Age_sin"] = np.sin(2 * np.pi * df["Age"] / 60.0)
            df["Age_cos"] = np.cos(2 * np.pi * df["Age"] / 60.0)

        # 3 ── sanity: any missing derived? create placeholder 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        # 4 ── robust outlier handling
        for col in ["Age", "PrismCoverTest"]:
            if col in df.columns:
                df[col] = _winsorize_series(df[col], k=iqr_k)

        # 5 ── noisy imputation
        rng = np.random.default_rng(42)
        for col in feature_cols:
            if df[col].isna().any():
                fill  = df[col].median() if df[col].dtype.kind != "f" else df[col].mean()
                sigma = df[col].std() if df[col].std() > 0 else 1.0
                noise = rng.normal(0, sigma * noise_scale, size=df[col].isna().sum())
                df.loc[df[col].isna(), col] = fill + noise

        # 6 ── extract numpy
        feats      = df[feature_cols].astype(np.float32).to_numpy()
        raw_labels = df[label_cols].astype(np.float32).to_numpy()
        labels     = (raw_labels > 0.0).astype(np.float32)  # dosage→binary

        # 7 ── standardize
        if scaler is None:
            scaler = StandardScaler()
        feats = scaler.fit_transform(feats) if is_train else scaler.transform(feats)
        self.scaler_  = scaler

        # 8 ── store tensor
        self._features_tensor = torch.as_tensor(feats,  dtype=dtype)
        self._labels_tensor   = torch.as_tensor(labels, dtype=dtype)

        # ----- 公共属性：方便其他模块使用 -----
        self.features = feats      # (N, d) np.float32
        self.labels   = labels     # (N, 8) np.float32

    # ----- torch Dataset API -----
    def __len__(self) -> int:
        return self._features_tensor.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._features_tensor[idx], self._labels_tensor[idx]
