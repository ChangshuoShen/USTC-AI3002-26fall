import numpy as np


class LinearRegressionModel:
    """用单个线性回归模型预测一个连续目标。"""

    def __init__(self, n_features: int):
        # 一个连续目标只需要一个权重向量和一个标量偏置。
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = np.float64(0.0)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        # 训练模型；features 为 (N, D)，targets 为 (N,)。
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        # 提示：使用矩阵乘法和偏置，返回形状为 (N,) 的预测值。
        raise NotImplementedError


class LogisticRegressionModel:
    """用逻辑回归从隐藏表征预测句子真假。"""

    def __init__(self, n_features: int):
        # 提示：单个二分类任务只需要一个权重向量和一个标量偏置。
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = np.float64(0.0)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        # 训练模型；features 为 (N, 2560)，targets 为 (N,) 的 0/1 标签。
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        # 提示：返回形状为 (N,) 的真类概率，不要在此处做 0.5 阈值化。"""
        raise NotImplementedError
