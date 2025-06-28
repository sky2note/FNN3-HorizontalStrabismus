#!/usr/bin/env python
"""
calibrate_temperature.py
────────────────────────────────────────────────────────
给定验证集 logits → 优化单一温度 T (温度缩放) 以最小化 NLL。
输出 JSON： { "T": ..., "brier": ..., "auc_macro": ... }
"""
from __future__ import annotations
import argparse, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

import config                                    # ← 统一列名
from dataset import StrabismusDataset            # ← 缺失 import 已补
from models.mlp_model import SurgeryIndicatorModel
from utils.metrics import compute_roc_auc

# ─── CLI ─────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",  required=True, help="验证集 CSV - 同 threshold_search 使用")
p.add_argument("--ckpt", required=True, help="模型权重 *.pth")
p.add_argument("--out",  default="calibration.json")
p.add_argument("--batch_size", type=int, default=256)
args = p.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1. 训练集 (fit scaler) + 验证集 (共享 scaler) ────────────
train_ds = StrabismusDataset(config.DEFAULT_CSV,   # 训练集 CSV
                             config.FEATURE_COLS,
                             config.LABEL_COLS,
                             is_train=True)
val_ds   = StrabismusDataset(args.csv,             # 验证集 CSV
                             config.FEATURE_COLS,
                             config.LABEL_COLS,
                             is_train=False,
                             scaler=train_ds.scaler_)

val_loader = DataLoader(val_ds, batch_size=args.batch_size)

# ─── 2. 加载模型权重 ─────────────────────────────────────────
model = SurgeryIndicatorModel(len(config.FEATURE_COLS),
                              hidden_dims=(32,),
                              dropout=0.5,
                              output_dim=len(config.LABEL_COLS)).to(DEVICE)
state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
model.load_state_dict(state); model.eval()

# ─── 3. 收集 logits / labels ────────────────────────────────
logits_list, labels_list = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        logits_list.append(model(xb.to(DEVICE)).cpu())
        labels_list.append(yb)
logits = torch.cat(logits_list)        # (N,8)
labels = torch.cat(labels_list).float()

# ─── 4. 优化温度 T  (LBFGS) ─────────────────────────────────
T = torch.ones(1, device=logits.device, requires_grad=True)
opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

def _nll():
    prob = torch.sigmoid(logits / T)
    return F.binary_cross_entropy(prob, labels)

opt.step(_nll)
T_opt = float(T.item())
print(f"★ 最佳温度 T = {T_opt:.3f}")

# ─── 5. 校准后评估 Brier & AUC ─────────────────────────────
prob_cal = torch.sigmoid(logits / T_opt).numpy()
labels_np = labels.numpy()
brier = float(np.mean((prob_cal - labels_np) ** 2))
auc   = float(compute_roc_auc(labels_np, prob_cal, average="macro"))
print(f"Brier={brier:.4f}  AUC={auc:.3f}")

# ─── 6. 保存结果 ───────────────────────────────────────────
out_json = {"T": T_opt, "brier": brier, "auc_macro": auc}
Path(args.out).write_text(json.dumps(out_json, indent=2))
print(f"✅  温度缩放结果写入 → {args.out}")
