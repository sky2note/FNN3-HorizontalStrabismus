# src/train/train_classification.py

"""
单折训练脚本 — ClassificationModel（新增 L2 正则 & OneCycleLR + ReduceLROnPlateau 调度）
----------------------------------------------------------
功能：
1. 加载单折训练/验证索引。
2. 构建 Dataset 和 DataLoader。
3. 实例化 ClassificationModel + 损失 + 优化器 + 双调度器 + EarlyStopping。
4. 训练并在验证集上评估，记录主要指标到日志。
5. OneCycleLR 在每个 batch 后调整学习率；ReduceLROnPlateau 在每个 epoch 后根据 val_loss 调整；EarlyStopping 监控 MCC 保存最优权重。
"""

import os
import pickle
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import C
from src.models.classification_model import ClassificationModel
from src.utils.data_utils import load_data
from src.utils.model_utils import EarlyStopping, set_seed
from src.utils.metrics import compute_classification_metrics


def train_fold(fold_id: int) -> None:
    # ──────────────── 0. 准备 ─────────────────
    set_seed(C.RANDOM_SEED + fold_id)

    model_dir = C.MODELS_DIR / "classification"
    log_dir   = C.LOG_DIR / "classification"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────── 1. 加载数据 ────────────────
    X, y_class, _ = load_data()
    with open(C.FOLDS_DIR / f"mskfold_{C.KFOLD}fold.pkl", "rb") as f:
        folds = pickle.load(f)
    train_idx = folds[fold_id]["train"]
    val_idx   = folds[fold_id]["val"]

    X_train, y_train = X[train_idx], y_class[train_idx]
    X_val,   y_val   = X[val_idx],   y_class[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,  num_workers=C.NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)

    # ──────────────── 2. 模型 & 组件 ────────────────
    device = C.DEVICE
    model = ClassificationModel().to(device)

    # 正负样本权重
    pos_counts = y_train.sum(axis=0)
    neg_counts = y_train.shape[0] - pos_counts
    pos_weight = torch.tensor(
        [(neg/pos) if pos>0 else 1.0 for neg,pos in zip(neg_counts, pos_counts)],
        dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 优化器：加 L2 正则
    optimizer = optim.Adam(
        model.parameters(),
        lr=C.LEARNING_RATE,
        weight_decay=1e-5
    )

    # OneCycleLR：在每个 batch 之后调用
    total_steps = C.NUM_EPOCHS * len(train_loader)
    cyc_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=C.LEARNING_RATE * 10,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=1e4,
        three_phase=False,
    )

    # ReduceLROnPlateau：在每个 epoch 之后调用
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=5,
        threshold=1e-3,
        cooldown=2,
        min_lr=1e-6
    )

    # 早停：监控 MCC
    early_stopper = EarlyStopping(
        patience=5,
        mode="max",
        delta=1e-4,
        save_path=model_dir / f"fold{fold_id}_best.pth",
    )

    # 日志文件头
    log_path = log_dir / f"fold{fold_id}_log.csv"
    with open(log_path, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss,train_mcc,val_mcc,val_auc,lr\n")

    # ──────────────── 3. 训练循环 ────────────────
    for epoch in range(1, C.NUM_EPOCHS + 1):
        # ——— 3.1 训练 ———
        model.train()
        running_loss = 0.0
        all_preds, all_targs = [], []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device).float(), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(Xb, return_logits=True)
            loss = criterion(logits, yb)
            loss.backward()
            if C.GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            optimizer.step()
            # 每个 batch 更新一次 OneCycleLR
            cyc_scheduler.step()

            running_loss += loss.item() * Xb.size(0)
            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_targs.append(yb.cpu())

        train_loss = running_loss / len(train_ds)
        train_preds = torch.vstack(all_preds).numpy()
        train_targs = torch.vstack(all_targs).numpy()
        train_mcc, train_auc = compute_classification_metrics(train_targs, train_preds)

        # ——— 3.2 验证 ———
        model.eval()
        val_loss = 0.0
        val_preds, val_targs = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device).float(), yb.to(device).float()
                logits = model(Xb, return_logits=True)
                loss = criterion(logits, yb)
                val_loss += loss.item() * Xb.size(0)
                val_preds.append(torch.sigmoid(logits).detach().cpu())
                val_targs.append(yb.cpu())

        val_loss /= len(val_ds)
        val_preds = torch.vstack(val_preds).numpy()
        val_targs = torch.vstack(val_targs).numpy()
        val_mcc, val_auc = compute_classification_metrics(val_targs, val_preds)

        # ——— 更新 ReduceLROnPlateau ———
        plateau_scheduler.step(val_loss)

        # ——— 记录 & 打印 ———
        current_lr = optimizer.param_groups[0]["lr"]
        with open(log_path, "a") as log_file:
            log_file.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},"
                           f"{train_mcc:.4f},{val_mcc:.4f},{val_auc:.4f},{current_lr:.6f}\n")
        print(f"[Fold {fold_id}][Epoch {epoch}] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"MCC: {val_mcc:.4f}, AUC: {val_auc:.4f}, LR: {current_lr:.2e}")

        # ——— 早停检查 ———
        early_stopper(val_mcc, model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Fold {fold_id} 完成，最佳模型保存在 {early_stopper.save_path}")


if __name__ == "__main__":
    for fold in range(C.KFOLD):
        train_fold(fold)
