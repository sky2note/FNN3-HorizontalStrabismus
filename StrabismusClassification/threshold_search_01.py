#!/usr/bin/env python
"""
threshold_search.py · 逐标签网格阈值搜索 (目标 = MCC)
────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse, json, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef as MCC

import config
from dataset import StrabismusDataset
from models.mlp_model import SurgeryIndicatorModel

# ─── CLI ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv-train", default=config.DEFAULT_CSV)
parser.add_argument("--csv-val",   default=config.DEFAULT_VAL)
parser.add_argument("--ckpt",      default=config.DEFAULT_CKPT)
parser.add_argument("--out",       default="best_thresholds.json")
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--hidden_dims", nargs="*", type=int, default=[64, 128, 64])
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--n_residual", type=int, default=2,   # ← 新增
                    help="ResidualFC 块数，需与训练保持一致")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1. scaler ───────────────────────────────────────
ds_train = StrabismusDataset(args.csv_train,
                             config.FEATURE_COLS, config.LABEL_COLS,
                             is_train=True)
ds_val   = StrabismusDataset(args.csv_val,
                             config.FEATURE_COLS, config.LABEL_COLS,
                             is_train=False, scaler=ds_train.scaler_)
val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

# ─── 2. 构造与加载模型 ───────────────────────────────
model = SurgeryIndicatorModel(
    input_dim=len(config.FEATURE_COLS),
    hidden_dims=tuple(args.hidden_dims),
    dropout=args.dropout,
    n_residual=args.n_residual,        # ← 保持一致
    output_dim=len(config.LABEL_COLS)
).to(DEVICE)

state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

# ─── 3. 收集概率 ────────────────────────────────────
with torch.no_grad():
    probs, trues = [], []
    for xb, yb in val_loader:
        probs.append(torch.sigmoid(model(xb.to(DEVICE))).cpu())
        trues.append(yb)
y_prob = torch.cat(probs).numpy()
y_true = torch.cat(trues).numpy()

# ─── 4. 网格阈值搜索 ────────────────────────────────
grid = np.arange(0.05, 1.00, 0.05)
best_thr, best_mcc = {}, {}
for i, lab in enumerate(config.LABEL_COLS):
    mccs = [MCC(y_true[:, i], (y_prob[:, i] >= t).astype(int)) for t in grid]
    j = int(np.argmax(mccs))
    best_thr[lab], best_mcc[lab] = float(grid[j]), float(mccs[j])
    print(f"{lab:10s}  best_thr={best_thr[lab]:.2f}   MCC={best_mcc[lab]:.3f}")

# ─── 5. 保存 ────────────────────────────────────────
Path(args.out).write_text(json.dumps(
    {"thresholds": best_thr, "mcc_perlabel": best_mcc}, indent=2))
print(f"\n✅  阈值 + MCC 写入 → {args.out}\n")
