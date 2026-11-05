"""课程提供：选择实验、构造模型、加载初值并保存结果。"""

import argparse
import time
from pathlib import Path

import torch

from models import MLP
from train import train
from utils import data as data_utils
from utils import plot, results
from utils.device import auto_device, check_device_available

# 每组依次为：隐藏层宽度、激活、更新次数、学习率。
PLANS = {
    "expression": {
        "A": ([16], "relu", 3000, 0.01),
        "B": ([16], "relu", 9000, 0.01),
        "C": ([64], "relu", 3000, 0.01),
        "D": ([8, 4], "relu", 3000, 0.01),
    },
    "gradient": {
        "A": ([32] * 2, "sigmoid", 1000, 0.05),
        "B": ([32] * 8, "sigmoid", 1000, 0.05),
        "C": ([32] * 8, "relu", 1000, 0.05),
        "D": ([32] * 8, "sigmoid", 1000, 0.1),
    },
}


def run(task, output, device="cpu"):
    # Output directory
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"请使用空的输出目录：{output}")
    output.mkdir(parents=True, exist_ok=True)

    # Device
    if device == "auto":
        device = auto_device()
    if not check_device_available(device):
        raise ValueError(f"设备 {device} 不可用。")
    
    # fix num threads
    torch.set_num_threads(1)

    # Load data and initial states
    data = data_utils.load_data(task, device)
    initial_states = data_utils.load_initial_states(task)

    optimizer = "adam" if task == "expression" else "sgd"
    rows, predictions, weights = [], {}, {}
    started = time.perf_counter()

    for group, (widths, activation, steps, learning_rate) in PLANS[task].items():
        # Model
        model = MLP(data["train_x"].shape[1], widths, activation)
        model.load_state_dict(initial_states[group], strict=True)
        model = model.to(device=device, dtype=torch.float32)

        # Train
        history = train(model, data["train_x"], data["train_y"], steps=steps,
                        learning_rate=learning_rate, optimizer=optimizer,
                        diagnostics=task == "gradient")

        # Evaluate
        with torch.no_grad():
            prediction = model(data["grid_x"])
            train_mse = float((model(data["train_x"]) - data["train_y"]).square().mean())
            grid_mse = float((prediction - data["grid_y"]).square().mean())
        rows.append({"group": group, "widths": widths, "activation": activation,
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "steps": steps, "learning_rate": learning_rate,
                     "train_mse": train_mse, "grid_mse": grid_mse, "history": history})
        predictions[group] = prediction.detach().cpu().numpy()
        weights[group] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        print(f"{task}/{group}: train={train_mse:.6g}, grid={grid_mse:.6g}", flush=True)

    report = {"task": task, "device": device, "torch": torch.__version__,
              "optimizer": optimizer, "batch_size": len(data["train_x"]),
              "data_file": f"assets/{task}.npz", "initial_states_file": "assets/initial_states.pt",
              "seconds": time.perf_counter() - started, "runs": rows}
    results.save_results(output, report, data, predictions, weights)
    plot.plot_results(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=PLANS, required=True)
    parser.add_argument("--output", type=Path, help="默认保存到 q1_neural_network/outputs/实验名")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "xpu"], default="auto")
    args = parser.parse_args()
    output = args.output if args.output is not None else Path(__file__).resolve().parent / "outputs" / args.task
    run(args.task, output, args.device)


if __name__ == "__main__":
    main()
