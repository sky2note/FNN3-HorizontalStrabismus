"""Data loading, winsorisation & scaling utilities (auto-rebuild scaler)."""
from __future__ import annotations
import math, os
from pathlib import Path
from typing import Tuple
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import C


def load_data(
    csv_path: Path | str | None = None,
    winsor_sigma: float = 2.5,
    scaler_path: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ① resolve paths
    csv_path = Path(csv_path or C.DATA_PATH).resolve()
    scaler_path = scaler_path or (C.SCALER_DIR / "standard.pkl")
    os.makedirs(scaler_path.parent, exist_ok=True)

    # ② read csv
    df = pd.read_csv(csv_path)

    # ③ fixed column order
    feats, classes, regs = C.FEATURE_COLS, C.CLASS_COLS, C.REG_COLS
    X_df       = df[list(feats)].copy()
    y_class_df = df[list(classes)].astype("int8")
    y_reg_df   = df[list(regs)].astype("float32")

    # ④ winsorise (optional)
    if winsor_sigma > 0:
        X_df = _winsorise_df(X_df, winsor_sigma)

    # ⑤ fit / load scaler  ──────────── NEW BLOCK ────────────
    rebuild = True
    if scaler_path.exists():
        scaler: StandardScaler = joblib.load(scaler_path)
        if getattr(scaler, "n_features_in_", -1) == len(feats):
            rebuild = False          # 维度匹配，可直接用
    if rebuild:
        scaler = StandardScaler().fit(X_df.values)
        joblib.dump(scaler, scaler_path)
    # ─────────────────────────────────────────────────────────

    X = scaler.transform(X_df.values).astype("float32")
    # return X, y_class_df.values, y_reg_df.values
    # 归一化到 0–1 区间，训练后再反缩放
    y_reg = (y_reg_df / C.TARGET_SCALE).values
    return X, y_class_df.values, y_reg

def _winsorise_df(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        mu, sd = out[col].mean(), out[col].std(ddof=0)
        if math.isclose(sd, 0.0):
            continue
        out[col] = out[col].clip(mu - sigma * sd, mu + sigma * sd)
    return out


__all__ = ["load_data"]
