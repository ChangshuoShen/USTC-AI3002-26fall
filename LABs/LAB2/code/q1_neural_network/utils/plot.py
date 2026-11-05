"""从保存结果绘图；也可单独运行 python -m utils.plot 输出目录。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def style(task, group, count):
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"] if task == "expression" else [
        "tab:blue", "tab:orange", "tab:green", "tab:red"]
    result = {"color": colors["ABCD".index(group)], "label": group, "linewidth": 1.8}
    # A/B 拟合曲线与 B/D 初始梯度会重合，用虚线和空心点显示。
    if group == ("A" if task == "expression" else "B"):
        result.update(linestyle="--", marker="o", markersize=4, markerfacecolor="white",
                      markevery=max(1, count // 12), zorder=5)
    return result


def save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_results(output):
    output = Path(output)
    report = json.loads((output / "results.json").read_text())
    task, rows = report["task"], report["runs"]
    figure, axis = plt.subplots(figsize=(7, 4))
    for row in rows:
        history = row["history"]
        loss = [point["train_mse"] if point["train_mse"] > 0 else np.nan for point in history]
        axis.plot([point["step"] for point in history], loss, **style(task, row["group"], len(history)))
        if any(point["train_mse"] == 0 for point in history):
            axis.annotate(f"{row['group']}: zero loss omitted on log scale", (0.03, 0.04), xycoords="axes fraction")
    axis.set(yscale="log", xlabel="Parameter updates", ylabel="Training MSE")
    axis.legend()
    axis.grid(alpha=0.25)
    save(figure, output / "training.png")

    if task == "expression":
        figure, axis = plt.subplots(figsize=(7, 4))
        with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
            grid, reference = arrays["grid"], arrays["reference"]
            section = np.isclose(grid[:, 1], 0)
            axis.plot(grid[section, 0], reference[section, 0], "k--", label="Target", linewidth=2)
            for row in rows:
                axis.plot(grid[section, 0], arrays[row["group"]][section, 0],
                          **style(task, row["group"], int(section.sum())))
        axis.set(title="Fitted section: x2 = 0", xlabel="x1", ylabel="Prediction")
        axis.legend()
        axis.grid(alpha=0.25)
        save(figure, output / "fitting.png")
    else:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for axis, timepoint, title in zip(axes, [0, -1], ["Initial", "After training"]):
            notes = []
            for row in rows:
                layers = row["history"][timepoint]["gradient_profile"]["layers"]
                ratios = [layer["relative_to_output"] for layer in layers]
                values = [value if value is not None and value > 0 else np.nan for value in ratios]
                axis.plot([layer["layer"] for layer in layers], values, **style(task, row["group"], len(layers)))
                if any(value is None or value == 0 for value in ratios):
                    notes.append(f"{row['group']}: zero/undefined values omitted")
            if notes:
                axis.text(0.03, 0.04, "\n".join(notes), transform=axis.transAxes, fontsize=8)
            axis.set(yscale="log", title=title, xlabel="Hidden layer", ylabel="Gradient RMS / output RMS")
            axis.tick_params(axis="y", labelleft=True)
            axis.legend()
            axis.grid(alpha=0.25)
        save(figure, output / "gradients.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    plot_results(parser.parse_args().output)
