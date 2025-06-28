# src/config.py


"""
Central configuration for the strabismus-surgery ML project.

Usage
-----
>>> from src.config import C
>>> C.DATA_PATH
PosixPath('.../data/Data_M1D6_Preproc.csv')
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# ─────────────────────────────────────────────
# 1. Dataclass holding all parameters
# ─────────────────────────────────────────────
@dataclass
class Config:
    # ---------- Paths ----------
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_PATH:   Path = PROJECT_ROOT / "data" / "Data_M1D6_Preproc.csv"
    FOLDS_DIR:   Path = PROJECT_ROOT / "data" / "folds"
    SCALER_DIR:  Path = PROJECT_ROOT / "data" / "scalers"
    MODELS_DIR:  Path = PROJECT_ROOT / "saved_models"
    LOG_DIR:     Path = PROJECT_ROOT / "logs"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"

    # ---------- Column names ----------
    FEATURE_COLS: tuple[str, ...] = (
        "Age",
        "PrismCoverTest",
        "AxL_mean",
        "AxL_diff",
        "SphEq_mean",
        "SphEq_diff",
    )
    CLASS_COLS: tuple[str, ...] = (
        "MRRsOD_Binary", "MRRsOS_Binary", "LRRsOD_Binary", "LRRsOS_Binary",
        "MRRcOD_Binary", "MRRcOS_Binary", "LRRcOD_Binary", "LRRcOS_Binary",
    )
    REG_COLS: tuple[str, ...] = (
        "MRRsOD", "MRRsOS", "LRRsOD", "LRRsOS",
        "MRRcOD", "MRRcOS", "LRRcOD", "LRRcOS",
    )

    # ---------- Network dims ----------
    INPUT_DIM: int = 6
    OUTPUT_DIM: int = 8
    HIDDEN_SIZES: tuple[int, ...] = (64, 32)   # new

    # ---------- Training hyper-params ----------
    DROPOUT: float = 0.3                       # new
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 500
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4                 # new
    KFOLD: int = 10
    RANDOM_SEED: int = 42
    NUM_WORKERS: int = 4

    USE_MIXED_PRECISION: bool = True           # AMP :contentReference[oaicite:2]{index=2}
    GRAD_CLIP: float | None = 1.0

    # ---------- Early-stopping ----------
    PATIENCE: int = 20
    LR_SCHED_FACTOR: float = 0.3                              ### NEW
    LR_SCHED_PATIENCE: int = 5

    # ---------- 目标缩放 ----------
    TARGET_SCALE: float = 10.0  # 0–10 mm → 0–1

    # ---------- CLS → REG mask ----------
    USE_CLASS_MASK: bool = False                     # new
    CLS_THRESHOLD: float = 0.5                 # new

    # ---------- Runtime fields ----------
    DEVICE: torch.device = field(init=False)

    # ───────── post-init: set device & seeds ─────────
    def __post_init__(self) -> None:
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_global_seed(self.RANDOM_SEED)

        # ensure folders exist
        for p in (
            self.FOLDS_DIR, self.SCALER_DIR,
            self.MODELS_DIR, self.LOG_DIR, self.RESULTS_DIR,
        ):
            os.makedirs(p, exist_ok=True)

    # allow runtime override
    def update(self, **kwargs: Dict[str, Any]) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # re-seed if needed
        if "RANDOM_SEED" in kwargs:
            self._set_global_seed(self.RANDOM_SEED)
        if "DEVICE" in kwargs and isinstance(kwargs["DEVICE"], str):
            self.DEVICE = torch.device(kwargs["DEVICE"])

    # reproducibility helper
    @staticmethod
    def _set_global_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   # deterministic conv. :contentReference[oaicite:3]{index=3}
        torch.backends.cudnn.benchmark = False      # disable autotune :contentReference[oaicite:4]{index=4}

# module-level singleton
C = Config()

__all__ = ["Config", "C"]


