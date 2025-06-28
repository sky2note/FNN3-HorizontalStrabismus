# src/train/bootstrap_evaluate.py


from __future__ import annotations
import json, numpy as np, torch
from pathlib import Path

from src.config import C
from src.utils.data_utils import load_data
from src.utils.metrics import masked_rmse_mm
from src.models.regression_model import RegressionModel

N_BOOT = 1000
RES_DIR = Path(C.RESULTS_DIR); RES_DIR.mkdir(parents=True, exist_ok=True)

# ── 载入 10 折模型 ──
models=[]
for k in range(C.KFOLD):
    m=RegressionModel().to(C.DEVICE)
    m.load_state_dict(torch.load(Path(C.MODELS_DIR)/f"reg_fold{k}.pth",
                                 map_location=C.DEVICE,
                                 weights_only=True), strict=False)
    m.eval(); models.append(m)

X, y_cls, y_reg = load_data(); N=len(X)
rng=np.random.default_rng(C.RANDOM_SEED); boot=[]
for _ in range(N_BOOT):
    idx=rng.integers(0,N,N)
    xb=torch.from_numpy(X[idx]).to(C.DEVICE)
    yb=torch.from_numpy(y_reg[idx]).to(C.DEVICE)
    mask=torch.from_numpy(y_cls[idx]).float().to(C.DEVICE)

    with torch.no_grad():
        pred=sum(m(xb) for m in models)/len(models)
    boot.append(masked_rmse_mm(pred.cpu(), yb.cpu(), mask.cpu()))

boot=np.array(boot)
ci=np.percentile(boot,[2.5,97.5])
print(f"Bootstrap RMSE {boot.mean():.3f} mm (95 % CI {ci[0]:.3f}–{ci[1]:.3f})")
json.dump({"mean":float(boot.mean()),"ci_low":float(ci[0]),"ci_high":float(ci[1])},
          open(RES_DIR/"bootstrap_summary.json","w"), indent=2)
np.save(RES_DIR/"bootstrap_rmse.npy", boot)
