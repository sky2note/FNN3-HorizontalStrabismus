#!/usr/bin/env python
"""
单折训练脚本（随机 8:2 划分），支持 FocalLoss / BCE 与 CC-Head
"""

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from dataset import StrabismusDataset
from models.mlp_model import SurgeryIndicatorModel
from utils.metrics import summary_all
from utils.losses import FocalLoss

# ───────────────────────── Utils ─────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_pos_weights(y: np.ndarray) -> torch.Tensor:
    pos = y.sum(axis=0); neg = y.shape[0] - pos
    return torch.as_tensor(neg / np.clip(pos, 1, None), dtype=torch.float32)

def init_sampler(y: np.ndarray) -> WeightedRandomSampler:
    w = 1.0 / np.clip(y.sum(axis=1), 1, None)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(w), replacement=True)

def run_epoch(model, loader, crit, dev, opt=None):
    training = opt is not None
    model.train() if training else model.eval()
    tot_loss, preds, labs = 0.0, [], []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            logit = model(xb); loss = crit(logit, yb)
            if training:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot_loss += loss.item() * xb.size(0)
            if not training:
                preds.append(torch.sigmoid(logit).cpu()); labs.append(yb.cpu())
    tot_loss /= len(loader.dataset)
    if training: return tot_loss, None
    pr, lb = torch.cat(preds).numpy(), torch.cat(labs).numpy()
    met = summary_all(lb, pr); met["loss"] = tot_loss; return tot_loss, met

# ───────────────────────── Main ─────────────────────────
def main(cfg):
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    print(f"⚙️  Device = {dev}")

    # 数据集
    ds_full = StrabismusDataset(cfg.csv, config.FEATURE_COLS, config.LABEL_COLS, is_train=True)
    idx = np.arange(len(ds_full)); np.random.default_rng(cfg.seed).shuffle(idx)
    split = int(0.8 * len(idx)); tr_idx, va_idx = idx[:split], idx[split:]

    ds_tr = StrabismusDataset(cfg.csv, config.FEATURE_COLS, config.LABEL_COLS, is_train=True)
    ds_va = StrabismusDataset(cfg.csv, config.FEATURE_COLS, config.LABEL_COLS,
                              is_train=False, scaler=ds_tr.scaler_)
    # 切片
    for dset, sub in [(ds_tr, tr_idx), (ds_va, va_idx)]:
        dset._features_tensor = dset._features_tensor[sub]
        dset._labels_tensor   = dset._labels_tensor[sub]
        dset.features, dset.labels = dset.features[sub], dset.labels[sub]

    tr_loader = DataLoader(ds_tr, batch_size=cfg.batch_size,
                           sampler=init_sampler(ds_tr.labels),
                           pin_memory=(dev.type == "cuda"))
    va_loader = DataLoader(ds_va, batch_size=cfg.batch_size, pin_memory=(dev.type == "cuda"))

    # 模型
    model = SurgeryIndicatorModel(
        input_dim=ds_tr._features_tensor.shape[1],
        hidden_dims=tuple(cfg.hidden_dims),
        dropout=cfg.dropout,
        n_residual=cfg.n_residual,
        output_dim=len(config.LABEL_COLS),
        use_cc=cfg.use_cc,
    ).to(dev)

    # 损失
    if cfg.loss == "focal":
        crit = FocalLoss(alpha=0.5, gamma=2.0).to(dev)
    else:
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=compute_pos_weights(ds_tr.labels).to(dev))

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_mcc, patience = -np.inf, 0
    for ep in range(1, cfg.epochs + 1):
        tl, _ = run_epoch(model, tr_loader, crit, dev, opt)
        vl, mt = run_epoch(model, va_loader, crit, dev)
        print(f"[{ep:03d}] train {tl:.4f} | val {vl:.4f} "
              f"AUC {mt['auc_macro']:.3f} MCC {mt['mcc_macro']:.3f}")
        if mt["mcc_macro"] > best_mcc + 1e-4:
            best_mcc, patience = mt["mcc_macro"], 0
            torch.save(model.state_dict(), cfg.out); print("  ↪ 🔥 save best")
        else:
            patience += 1
            if patience >= cfg.patience: print("⏹  Early stop"); break

# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--csv", default=config.DEFAULT_CSV)
    pa.add_argument("--out", default="best_model.pth")
    pa.add_argument("--batch_size", type=int, default=32)
    pa.add_argument("--epochs", type=int, default=300)
    pa.add_argument("--hidden_dims", nargs="*", default=[64, 128, 64], type=int)
    pa.add_argument("--dropout", type=float, default=0.3)
    pa.add_argument("--n_residual", type=int, default=2)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--patience", type=int, default=20)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--cpu", action="store_true")
    pa.add_argument("--loss", choices=["focal", "bce"], default="focal")
    pa.add_argument("--use_cc", action="store_true",
                    help="启用 Classifier-Chain Head")
    cfg = pa.parse_args(); main(cfg)
