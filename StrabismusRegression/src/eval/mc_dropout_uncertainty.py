"""
MC‐Dropout predictive mean & std for a single fold model.
$ python -m src.eval.mc_dropout_uncertainty --fold 0 --T 50
"""

import argparse, json
import numpy as np, torch
from tqdm import trange

from src.config import C
from src.utils.data_utils import load_data
from src.models.regression_model import RegressionModel

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--T",    type=int, default=50, help="MC samples")
    p.add_argument("--outfile", default="mc_dropout_fold0.npy")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse()
    X, _, _ = load_data()
    model = RegressionModel().to(C.DEVICE)
    state = torch.load(C.MODELS_DIR / f"fold{args.fold}.pt", map_location=C.DEVICE)
    model.load_state_dict(state)
    model.train()                           # 关键：开启 dropout
    preds = []

    for _ in trange(args.T):
        preds.append(model(torch.tensor(X, dtype=torch.float32, device=C.DEVICE)).cpu().numpy())

    preds = np.stack(preds, axis=0)         # (T, N, 8)
    mean  = preds.mean(0)
    std   = preds.std(0)                    # 按样本方差评估不确定度

    np.savez(C.RESULTS_DIR / args.outfile, mean=mean, std=std)
    print("✅ MC‐Dropout 结果保存 →", C.RESULTS_DIR)

if __name__ == "__main__":
    main()
