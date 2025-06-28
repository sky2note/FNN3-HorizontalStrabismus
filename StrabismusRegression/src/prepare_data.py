"""
src/prepare_data.py
===================

功能：
1. 调用 utils.data_utils.load_data() 读取并预处理数据。
2. 使用 MultilabelStratifiedKFold 将样本按多标签分层方式
   划分为 10 折，保存每折的 train / val 索引。
3. 额外生成 1000 份 bootstrap 采样索引，可供稳健评估使用。

运行
----
$ python -m src.prepare_data
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # pip install iterative-stratification
from tqdm import tqdm

from src.config import C
from src.utils.data_utils import load_data


def main(
    n_splits: int = C.KFOLD,
    n_bootstrap: int = 1000,
    seed: int = C.RANDOM_SEED,
) -> None:
    # ─────────────────────── 1. 读取数据 ───────────────────────
    X, y_class, _ = load_data()     # y_class 用于分层；X 仅拿样本数

    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)

    # ─────────────────────── 2. 生成 K 折索引 ───────────────────
    folds_dir = C.FOLDS_DIR
    os.makedirs(folds_dir, exist_ok=True)

    cv = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )  # 论文算法参见 Sechidis et al. 2011 :contentReference[oaicite:0]{index=0}

    fold_indices: list[dict[str, np.ndarray]] = []
    for fold_id, (train_idx, val_idx) in enumerate(cv.split(np.zeros(n_samples), y_class)):
        fold_indices.append({"train": train_idx, "val": val_idx})

    with open(folds_dir / f"mskfold_{n_splits}fold.pkl", "wb") as f:
        pickle.dump(fold_indices, f)
    print(f"✅ 已保存 {n_splits} 折索引 → {folds_dir}")

    # ─────────────────────── 3. 生成 Bootstrap 索引 ─────────────
    if n_bootstrap > 0:
        boot_list = [
            rng.choice(n_samples, size=n_samples, replace=True) for _ in tqdm(range(n_bootstrap), desc="Bootstrap")
        ]
        with open(folds_dir / f"bootstrap_{n_bootstrap}.pkl", "wb") as f:
            pickle.dump(boot_list, f)
        print(f"✅ 已保存 {n_bootstrap} 份 bootstrap 索引 → {folds_dir}")


if __name__ == "__main__":
    main()
