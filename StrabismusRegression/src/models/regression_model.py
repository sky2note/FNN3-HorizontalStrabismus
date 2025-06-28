# src/models/regression_model.py

"""
Multi-output regression MLP for 8 surgical doses (0–10 mm).
"""
from __future__ import annotations
import torch, torch.nn as nn
from src.config import C


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        in_dim   = getattr(C, "INPUT_DIM", 6)
        hid_dims = list(getattr(C, "HIDDEN_SIZES", (64, 32)))
        out_dim  = getattr(C, "OUTPUT_DIM", 8)
        dropout  = getattr(C, "DROPOUT", 0.0)

        dims = [in_dim] + hid_dims + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]),
                       nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))  # linear output

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = torch.clamp(self.net(x), 0.0, 10.0)  # constrain to 0-10 mm
        return y * mask if mask is not None else y
