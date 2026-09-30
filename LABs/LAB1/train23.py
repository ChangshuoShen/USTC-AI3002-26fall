"""Train the coordinate or truth probe and save its parameters."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model import LinearRegressionModel, LogisticRegressionModel


# 加载 task 任务的 train/valid 数据集，特征取自 layer 层
def load_split(task: str, split: str, layer: int) -> tuple[np.ndarray, np.ndarray]:

    # 在 Python 中，Path(__file__) 表示当前脚本文件的路径
    data_dir = Path(__file__).parent / "datasets" / task

    features = np.load(data_dir / f"{split}.layer{layer}.npy", allow_pickle=False)
    labels = pd.read_csv(data_dir / f"{split}.csv")
    if task == "world":
        targets = labels[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        # 形状为 (N, 2)
    else:
        targets = labels["label"].to_numpy(dtype=np.float64)
        # 形状为 (N,)
    if len(features) != len(targets):
        raise ValueError("The feature and label files have different row counts")

    return features.astype(np.float64), targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("world", "truth"))
    parser.add_argument("layer", type=int)
    parser.add_argument("output", type=Path, help="Path to the output .npz file")
    # 你可以在此处加入更多的命令行参数，例如学习率、训练轮数等
    args = parser.parse_args()

    # 加载数据
    train_x, train_y = load_split(args.task, "train", args.layer)
    valid_x, valid_y = load_split(args.task, "valid", args.layer)

    if args.task == "world":
        # 纬度和经度各训练一个一维线性回归模型。
        latitude_targets = train_y[:, 0]
        longitude_targets = train_y[:, 1]

        latitude_model = LinearRegressionModel(train_x.shape[1])
        longitude_model = LinearRegressionModel(train_x.shape[1])
        latitude_model.fit(train_x, latitude_targets)
        longitude_model.fit(train_x, longitude_targets)

        latitude_predictions = latitude_model.predict(valid_x)
        longitude_predictions = longitude_model.predict(valid_x)
        prediction = np.column_stack((latitude_predictions, longitude_predictions))
        weights = np.stack((latitude_model.weights, longitude_model.weights))
        bias = np.array((latitude_model.bias, longitude_model.bias))
    else:
        model = LogisticRegressionModel(train_x.shape[1])
        model.fit(train_x, train_y)
        prediction = model.predict(valid_x)
        weights = model.weights
        bias = model.bias

    if prediction.shape != valid_y.shape:
        raise ValueError(f"Expected predictions with shape {valid_y.shape}")

    if args.task == "world":
        print(f"Validation MSE: {np.mean((prediction - valid_y) ** 2):.4f}")
    else:
        print(f"Validation accuracy: {np.mean((prediction >= 0.5) == valid_y):.4f}")

    # 创造输出目录（如果不存在的话）
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 输出文件的名称被 output 参数指定，文件格式为 .npz
    np.savez(
        args.output,
        weights=np.asarray(weights, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
    )
    print(f"Saved {args.output} (layer {args.layer})")


if __name__ == "__main__":
    main()
