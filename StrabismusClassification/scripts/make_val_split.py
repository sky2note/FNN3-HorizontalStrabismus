# scripts/make_val_split.py
import pandas as pd, numpy as np, argparse, pathlib, json, os

p = argparse.ArgumentParser()
p.add_argument("--in",  default="data/Data_M1D5_Preproc.csv")
p.add_argument("--out", default="data/val.csv")
p.add_argument("--val_frac", type=float, default=0.2)
p.add_argument("--seed", type=int, default=42)
cfg = p.parse_args()

df = pd.read_csv(cfg.__dict__["in"])
rng = np.random.default_rng(cfg.seed)
val_idx = rng.permutation(len(df))[:int(len(df)*cfg.val_frac)]
df.iloc[val_idx].to_csv(cfg.out, index=False)
print(f"✅  验证集保存 {cfg.out}   (rows={len(val_idx)})")
