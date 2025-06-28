# src/utils/model_utils.py


"""Reproducibility, early stopping, checkpoint helpers."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch


# ───────────────── reproducibility ─────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ───────────────── early stopping ─────────────────
class EarlyStopping:
    def __init__(
        self, patience: int = 30, mode: str = "min",
        delta: float = 0.0, save_path: str | Path = "checkpoint.pth"
    ) -> None:
        assert mode in {"min", "max"}
        self.patience, self.mode, self.delta = patience, mode, delta
        self.save_path = Path(save_path)
        self.counter, self.best_score = 0, None
        self.early_stop = False
        self._better = (lambda cur, best: (best - cur) > delta) if mode == "min" \
            else (lambda cur, best: (cur - best) > delta)

    def __call__(self, current: float, model: torch.nn.Module) -> None:
        if self.best_score is None or self._better(current, self.best_score):
            self.best_score, self.counter = current, 0
            self._save(model)
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

    def _save(self, model: torch.nn.Module) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.save_path)

    def load_best_model(self, model: torch.nn.Module) -> None:
        state_dict = torch.load(self.save_path, map_location="cpu", weights_only=True)  # UPDATED
        model.load_state_dict(state_dict, strict=False)


# ───────────────── checkpoint wrappers ─────────────────
def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, map_location=None) -> None:
    state_dict = torch.load(path, map_location=map_location, weights_only=True)         # UPDATED
    model.load_state_dict(state_dict, strict=False)
