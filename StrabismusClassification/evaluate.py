#!/usr/bin/env python
"""
evaluate.py
──────────────────────────────────────────────────────────
1. 读取模型权重 (--ckpt)
2. 读取温度标定文件 (--temp)
3. 读取逐标签阈值文件 (--thr)
4. 在测试集 (--csv-test) 上计算多标签指标并保存 JSON
"""
from __future__ import annotations
import argparse, json, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader

import config
from dataset import StrabismusDataset
from models.mlp_model import SurgeryIndicatorModel
from utils.metrics import summary_all, compute_roc_auc, compute_mcc, compute_f1

# ─── CLI ────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    "Evaluate model with calibrated temperature + custom thresholds"
)
parser.add_argument("--csv-test", default="data/test.csv")
parser.add_argument("--csv-train", default=config.DEFAULT_CSV)
parser.add_argument("--ckpt", default="best_model.pth")
parser.add_argument("--thr",  default="best_thresholds.json")
parser.add_argument("--temp", default="calibration.json")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--out", default="eval_metrics.json")

# 与训练保持一致的网络结构
parser.add_argument("--hidden_dims", nargs="*", type=int, default=[64, 128, 64])
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--n_residual", type=int, default=2,
                    help="ResidualFC 块数，必须与训练时一致")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1. 加载阈值 & 温度 ───────────────────────────────
thr_info = json.load(open(args.thr))
THR_VEC  = np.array([thr_info["thresholds"][lab] for lab in config.LABEL_COLS])

temp_info = json.load(open(args.temp))
T_raw = temp_info["T"]
if isinstance(T_raw, list):
    T_vec = torch.tensor(T_raw, dtype=torch.float32, device=DEVICE)
    apply_temp = lambda logits: torch.sigmoid(logits / T_vec)
else:
    T_opt = float(T_raw)
    apply_temp = lambda logits: torch.sigmoid(logits / T_opt)

# ─── 2. DataLoader ────────────────────────────────────
train_ds = StrabismusDataset(args.csv_train,
                             config.FEATURE_COLS, config.LABEL_COLS,
                             is_train=True)
test_ds  = StrabismusDataset(args.csv_test,
                             config.FEATURE_COLS, config.LABEL_COLS,
                             is_train=False, scaler=train_ds.scaler_)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# ─── 3. 构造并加载模型 ────────────────────────────────
model = SurgeryIndicatorModel(
    input_dim=len(config.FEATURE_COLS),
    hidden_dims=tuple(args.hidden_dims),
    dropout=args.dropout,
    n_residual=args.n_residual,     # ← 保证与训练一致
    output_dim=len(config.LABEL_COLS)
).to(DEVICE)

state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

# ─── 4. 推理 ─────────────────────────────────────────
with torch.no_grad():
    prob_list, true_list = [], []
    for xb, yb in test_loader:
        logits = model(xb.to(DEVICE))
        prob   = apply_temp(logits).cpu()
        prob_list.append(prob)
        true_list.append(yb)
y_prob = torch.cat(prob_list).numpy()   # (N,8)
y_true = torch.cat(true_list).numpy()
y_pred = (y_prob >= THR_VEC).astype(int)

# ─── 5. 指标计算 ────────────────────────────────────
metrics = summary_all(y_true, y_prob, y_pred)
metrics["auc_perlabel"] = compute_roc_auc(y_true, y_prob, average=None).tolist()
metrics["mcc_perlabel"] = compute_mcc(y_true, y_pred, average=None).tolist()
metrics["f1_perlabel"]  = compute_f1(y_true, y_pred, average=None).tolist()

# ─── 6. 保存结果 ────────────────────────────────────
Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print("\n=== 评估完成 ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k:12s}: {v:.4f}")
print(f"\n✅  详细指标写入 → {args.out}\n")
