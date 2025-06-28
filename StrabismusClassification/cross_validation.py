#!/usr/bin/env python
"""
10-Fold CV  +  保存每折 y_true / y_prob
支持 FocalLoss 与可选 Classifier-Chain Head (--use_cc)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, matthews_corrcoef

import config
from dataset import StrabismusDataset
from models.mlp_model import SurgeryIndicatorModel
from utils.metrics import summary_all
from utils.losses import FocalLoss

# ────────── CLI ──────────
p = argparse.ArgumentParser()
p.add_argument("--csv", default=config.DEFAULT_CSV)
p.add_argument("--out-dir", type=Path, default="results/cv")
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--hidden_dims", nargs="*", type=int, default=[64, 128, 64])
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--patience", type=int, default=8)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--cpu", action="store_true")
p.add_argument("--use_cc", action="store_true", help="启用 Classifier-Chain Head")
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
args.out_dir.mkdir(parents=True, exist_ok=True)

FEATS, LABELS = config.FEATURE_COLS, config.LABEL_COLS

# ────────── helpers ──────────
@torch.no_grad()
def eval_loader(model, loader):
    model.eval(); ys, ps = [], []
    for xb, yb in loader:
        ys.append(yb); ps.append(torch.sigmoid(model(xb.to(DEVICE))).cpu())
    y_true, y_prob = torch.cat(ys).numpy(), torch.cat(ps).numpy()

    summ = summary_all(y_true, y_prob)

    # 逐标签 AUC / MCC 重新计算
    summ["auc_per"] = roc_auc_score(y_true, y_prob, average=None)
    pred = (y_prob >= 0.5).astype(int)
    summ["mcc_per"] = np.array(
        [matthews_corrcoef(y_true[:, i], pred[:, i]) for i in range(y_true.shape[1])]
    )
    return summ, y_true, y_prob

def train_fold(ds, tr_idx, va_idx, fold):
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(Subset(ds, va_idx), batch_size=args.batch_size)

    model = SurgeryIndicatorModel(
        len(FEATS), tuple(args.hidden_dims),
        args.dropout, 2, len(LABELS),
        use_cc=args.use_cc
    ).to(DEVICE)

    crit = FocalLoss(alpha=0.5, gamma=2.0).to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc, wait, best_met = -np.inf, 0, None
    for _ in range(args.epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        met, _, _ = eval_loader(model, va_loader)
        if met["auc_macro"] > best_auc + 1e-4:
            best_auc, best_met, wait = met["auc_macro"], met, 0
        else:
            wait += 1
            if wait >= args.patience: break

    # 最佳模型验证集概率保存
    _, y_true, y_prob = eval_loader(model, va_loader)
    np.savez(args.out_dir / f"fold{fold}_val_preds.npz",
             y_true=y_true, y_prob=y_prob)

    print(f"[Fold {fold}] AUC={best_met['auc_macro']:.3f}  MCC={best_met['mcc_macro']:.3f}")
    return best_met

# ────────── main ──────────
def main():
    ds = StrabismusDataset(args.csv, FEATS, LABELS, is_train=True)
    mskf = MultilabelStratifiedKFold(10, shuffle=True, random_state=args.seed)

    macro_rows, auc_rows, mcc_rows = [], [], []
    for fold, (tr, va) in enumerate(mskf.split(np.zeros(len(ds)), ds.labels)):
        m = train_fold(ds, tr, va, fold); m["fold"] = fold
        macro_rows.append({k: v for k, v in m.items() if k not in {"auc_per", "mcc_per"}})
        auc_rows.append(m["auc_per"]); mcc_rows.append(m["mcc_per"])

    macro_df = pd.DataFrame(macro_rows)
    macro_df.to_csv(args.out_dir / "cv_metrics_macro.csv", index=False)

    per_df = pd.DataFrame(auc_rows, columns=[f"AUC_{l}" for l in LABELS])
    per_df = per_df.join(pd.DataFrame(mcc_rows, columns=[f"MCC_{l}" for l in LABELS]))
    per_df.insert(0, "fold", range(10))
    per_df.to_csv(args.out_dir / "cv_metrics_perlabel.csv", index=False)

    summary = macro_df.mean(numeric_only=True).to_dict()
    (args.out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print("=== CV 完成 ==="); print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
