"""
bootstrap_eval.py
-----------------
在完整训练集上反复自举 (B=1000) 评估 MLP，输出 95% 置信区间。
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from dataset import StrabismusDataset
from models.mlp_model import SurgeryIndicatorModel
from utils.metrics import macro_micro_metrics

# ---------------- 参数 ----------------
try:
    import config
    LR, BATCH_SIZE, NUM_EPOCHS = config.LR, config.BATCH_SIZE, config.NUM_EPOCHS
    DROPOUT_P, HIDDEN_UNITS = config.DROPOUT_P, config.HIDDEN_UNITS
except (ImportError, AttributeError):
    LR, BATCH_SIZE, NUM_EPOCHS = 1e-3, 32, 50
    DROPOUT_P, HIDDEN_UNITS = 0.5, 32

B = 1000                      # 自举次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("results/bootstrap")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_on_subset(idx):
    ds = StrabismusDataset(train=True)
    loader = DataLoader(Subset(ds, idx), batch_size=BATCH_SIZE, shuffle=True)

    # 训练一个迷你模型（少 epoch 即可）
    pos_freq   = ds.labels[idx].mean(dim=0)
    pos_weight = ((1.0 - pos_freq) / (pos_freq + 1e-6)).to(DEVICE)
    model = SurgeryIndicatorModel(HIDDEN_UNITS, DROPOUT_P).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim     = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(NUM_EPOCHS):           # 不做验证早停，简单跑固定轮
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
    return model


@torch.no_grad()
def predict_full(model, full_loader):
    """对全量数据预测概率"""
    model.eval()
    prob = []
    for xb, _ in full_loader:
        xb = xb.to(DEVICE)
        prob.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.vstack(prob)


def main():
    full_ds = StrabismusDataset(train=True)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False)

    rng = np.random.default_rng(42)
    metrics_list = []

    for b in range(B):
        # 自举索引
        boot_idx = rng.choice(len(full_ds), size=len(full_ds), replace=True)
        model = train_on_subset(boot_idx)
        y_prob = predict_full(model, full_loader)
        y_true = full_ds.labels.numpy()
        m = macro_micro_metrics(y_true, y_prob)
        metrics_list.append(m)

        if (b + 1) % 50 == 0:
            print(f"Bootstrap {b+1}/{B}")

    df = pd.DataFrame(metrics_list)
    df.to_csv(OUT_DIR / "bootstrap_metrics.csv", index=False)

    # 95% 置信区间
    summary = {
        k: {
            "mean": df[k].mean(),
            "ci95_lower": df[k].quantile(0.025),
            "ci95_upper": df[k].quantile(0.975),
        }
        for k in ["macro_auc", "macro_mcc", "macro_f1"]
    }
    with open(OUT_DIR / "bootstrap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Bootstrap 完成 ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
