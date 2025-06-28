#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 读取净获益数据
# 假设已有 CSV: threshold, net_benefit, treat_all, treat_none
df = pd.read_csv(Path(__file__).parents[1] / "results" / "decision_curve_data.csv")

# 2. 绘图
plt.figure(figsize=(6, 4))
plt.plot(df["threshold"], df["net_benefit"], label="Model")
plt.plot(df["threshold"], df["treat_all"], "--", label="Treat All")
plt.plot(df["threshold"], df["treat_none"], ":", label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis")
plt.legend()
plt.tight_layout()
out = Path(__file__).parents[1] / "results" / "figs" / "decision_curve.png"
plt.savefig(out, dpi=300)
plt.close()
print(f"Saved decision curve to {out}")
