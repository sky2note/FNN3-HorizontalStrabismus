# src/train/train_regression.py


"""

Train multi-output regression model for one CV fold.

Run:
$ python -m src.train.train_regression --fold 0
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch import amp                              # 新 AMP API
from sklearn.model_selection import KFold

from src.config import C
from src.utils.data_utils import load_data
from src.utils.model_utils import set_seed, EarlyStopping
from src.models.regression_model import RegressionModel

# ───────────────────────────────────────────────
# 预计算 8 个手术标签的阳性权重  w = 1 / p
# ───────────────────────────────────────────────
_, y_cls_all, _ = load_data()
POS_RATE   = y_cls_all.mean(0)                       # (8,)
CLS_WEIGHTS = torch.as_tensor(1.0 / POS_RATE, dtype=torch.float32)


# ───────────────────────────────────────────────
# 辅助：根据索引构造 DataLoader
# ───────────────────────────────────────────────
def _make_loader(X, y_reg, y_cls, idx, train: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X[idx]),
                       torch.from_numpy(y_reg[idx]),
                       torch.from_numpy(y_cls[idx]))
    return DataLoader(ds,
                      batch_size=C.BATCH_SIZE,
                      shuffle=train,
                      num_workers=C.NUM_WORKERS,
                      pin_memory=True)


# ───────────────────────────────────────────────
# 训练单折
# ───────────────────────────────────────────────
def train_fold(fold: int) -> RegressionModel:
    set_seed(C.RANDOM_SEED + fold)

    # 1) 数据切分
    X, y_cls, y_reg = load_data()
    kf = KFold(n_splits=C.KFOLD, shuffle=True, random_state=C.RANDOM_SEED)
    train_idx, val_idx = list(kf.split(X))[fold]
    train_loader = _make_loader(X, y_reg, y_cls, train_idx, train=True)
    val_loader   = _make_loader(X, y_reg, y_cls, val_idx,   train=False)

    # 2) 模型 + 优化器 + 调度器
    device = torch.device(C.DEVICE)
    model  = RegressionModel().to(device)
    optim  = torch.optim.Adam(model.parameters(),
                              lr=C.LEARNING_RATE,
                              weight_decay=C.WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)
    scheduler = ReduceLROnPlateau(
        optim, mode="min",
        factor=C.LR_SCHED_FACTOR,
        patience=C.LR_SCHED_PATIENCE)

    scaler = amp.GradScaler(enabled=C.USE_MIXED_PRECISION)

    stopper = EarlyStopping(
        patience=C.PATIENCE,
        mode="min",
        save_path=Path(C.MODELS_DIR) / f"reg_fold{fold}.pth")

    # 3) 训练循环
    for epoch in range(1, C.NUM_EPOCHS + 1):
        # —— Train —— #
        model.train()
        train_loss = 0.0
        for xb, yb_reg, yb_cls in train_loader:
            xb, yb_reg, yb_cls = xb.to(device), yb_reg.to(device), yb_cls.to(device)
            mask     = yb_cls.float()                     # (B,8)
            weights  = CLS_WEIGHTS.to(device)            # (8,)

            with amp.autocast(device_type="cuda", enabled=C.USE_MIXED_PRECISION):
                preds = model(xb)                        # already 0-1 scale
                loss_elem = criterion(preds, yb_reg) * mask * weights
                loss = loss_elem.sum() / (mask * weights).sum()

            scaler.scale(loss).backward()
            if C.GRAD_CLIP:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            train_loss += loss.item() * xb.size(0)

        # —— Validate —— #
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), amp.autocast(device_type="cuda", enabled=C.USE_MIXED_PRECISION):
            for xb, yb_reg, yb_cls in val_loader:
                xb, yb_reg, yb_cls = xb.to(device), yb_reg.to(device), yb_cls.to(device)
                mask    = yb_cls.float()
                weights = CLS_WEIGHTS.to(device)
                preds   = model(xb)
                loss_elem = criterion(preds, yb_reg) * mask * weights
                loss = loss_elem.sum() / (mask * weights).sum()
                val_loss += loss.item() * xb.size(0)

        n_train, n_val = len(train_idx), len(val_idx)
        avg_train = train_loss / n_train
        avg_val   = val_loss   / n_val
        print(f"[Fold {fold}] Epoch {epoch:03d} | train {avg_train:.4f} | val {avg_val:.4f}")

        scheduler.step(avg_val)
        stopper(avg_val, model)
        if stopper.early_stop:
            print(f"[Fold {fold}] -- early stop @ {epoch}")
            break

    stopper.load_best_model(model)
    # ── 保存本折验证索引，供 cross_validation.py 读取 ──
    fold_dir = Path(C.FOLDS_DIR); fold_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(fold_dir / f"fold{fold}_val.txt", val_idx, fmt="%d")

    return model, stopper.best_score


# ───────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0,
                        help="0-based fold index")
    args = parser.parse_args()
    train_fold(args.fold)
