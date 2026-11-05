"""保存指标、训练历史、网格预测和最终模型参数。"""

import csv
import json

import numpy as np
import torch


def save_results(output, report, data, predictions, weights):
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = ["group", "parameters", "steps", "learning_rate", "train_mse", "grid_mse"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["runs"])
    np.savez_compressed(output / "predictions.npz", grid=data["grid_x"].cpu().numpy(),
                        reference=data["grid_y"].cpu().numpy(), **predictions)
    torch.save(weights, output / "models.pt")
