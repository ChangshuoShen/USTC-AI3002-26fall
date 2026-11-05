"""读取 assets 中已固定的数据与初始参数，不在训练时重新采样。"""

from pathlib import Path

import numpy as np
import torch


ASSETS = Path(__file__).resolve().parents[1] / "assets"


def load_data(task, device="cpu"):
    with np.load(ASSETS / f"{task}.npz", allow_pickle=False) as arrays:
        return {name: torch.from_numpy(arrays[name]).to(device=device)
                for name in ("train_x", "train_y", "grid_x", "grid_y")}


def load_initial_states(task):
    states = torch.load(ASSETS / "initial_states.pt", map_location="cpu", weights_only=True)
    return states[task]
