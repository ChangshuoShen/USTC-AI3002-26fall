"""读取实验配置和固定特征，准备拟合数据与公共评价子集。"""
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from utils.runtime import DTYPES

EVALUATION_FIELDS = {"selection_metric", "silhouette_sample_size", "silhouette_seed", "threads"}

METRIC_SPECS = {
    "silhouette": dict(function="silhouette_score", direction="maximize", scope="subset", bounds=(-1., 1.)),
    "calinski_harabasz": dict(function="calinski_harabasz_score", direction="maximize", scope="full", bounds=(0., None)),
    "davies_bouldin": dict(function="davies_bouldin_score", direction="minimize", scope="full", bounds=(0., None)),
}
METRICS = tuple(METRIC_SPECS)


def validate_config(value, *, development=False):
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    config = deepcopy(value)
    config.setdefault("selection_metric", "silhouette")
    config.setdefault("dtype", "float32")
    if config["dtype"] not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    required = ("candidate_k", "seeds", "n_init", "kmeans_max_iter", "kmeans_tol",
                "gmm_max_iter", "gmm_tol", "reg_covar", "standardize",
                "silhouette_sample_size", "silhouette_seed", "threads")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing configuration fields: {missing}")
    for field in ("candidate_k", "seeds"):
        items = config[field]
        if (not isinstance(items, list) or len(items) < 2
                or any(type(x) is not int for x in items) or len(set(items)) != len(items)):
            raise ValueError(f"{field} must contain at least two distinct integers")
    if any(not (1 if development else 5) <= k <= 30 for k in config["candidate_k"]):
        raise ValueError("candidate K outside allowed range")
    if any(s < 0 for s in config["seeds"]):
        raise ValueError("random seeds must be nonnegative")
    for field in ("n_init", "kmeans_max_iter", "gmm_max_iter", "threads"):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("kmeans_tol", "gmm_tol", "reg_covar"):
        v = config[field]
        if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    if config["reg_covar"] == 0:
        raise ValueError("reg_covar must be positive")
    if type(config["standardize"]) is not bool:
        raise ValueError("standardize must be a boolean")
    if type(config["silhouette_seed"]) is not int or config["silhouette_seed"] < 0:
        raise ValueError("silhouette_seed must be a nonnegative integer")
    if type(config["silhouette_sample_size"]) is not int or config["silhouette_sample_size"] < 2:
        raise ValueError("silhouette_sample_size must be an integer >= 2")
    if config["selection_metric"] not in METRICS:
        raise ValueError(f"selection_metric must be one of {METRICS}")
    return config


def load_features(path, *, development=False):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"X"}:
            raise ValueError("input NPZ must contain only X")
        X = archive["X"]
    if development:
        if (X.ndim != 2 or min(X.shape) < 1 or not np.issubdtype(X.dtype, np.floating)
                or not np.isfinite(X).all()):
            raise ValueError("development input requires a finite floating-point (N,D) array")
    elif X.shape != (56000, 512) or X.dtype != np.float32 or not np.isfinite(X).all():
        raise ValueError("expected finite float32 X with shape (56000,512)")
    return X


def prepare_data(path, config, *, development=False):
    X = load_features(path, development=development)
    if max(config["candidate_k"]) > len(X):
        raise ValueError("candidate K cannot exceed sample count")
    if config["silhouette_sample_size"] > len(X):
        raise ValueError("silhouette_sample_size cannot exceed sample count")
    std = X.std(axis=0, dtype=np.float64)
    mean = X.mean(axis=0, dtype=np.float64) if config["standardize"] else np.zeros(X.shape[1])
    scale = std.copy() if config["standardize"] else np.ones(X.shape[1])
    scale[scale == 0] = 1
    # 预处理统计量用双精度计算，最后再转为拟合精度；不改变原始行序。
    Z = ((X.astype(np.float64) - mean) / scale).astype(config.get("dtype", "float32"))
    if not np.isfinite(Z).all():
        raise ValueError("preprocessing produced non-finite values")
    # 所有候选共用同一评价子集，避免抽样差异影响比较。
    sample = np.sort(np.random.default_rng(config["silhouette_seed"]).choice(
        len(X), config["silhouette_sample_size"], replace=False))
    summary = dict(shape=list(X.shape), dtype=str(X.dtype), compute_dtype=str(Z.dtype), finite=True,
                   std_min=float(std.min()), std_max=float(std.max()),
                   constant_dimensions=int(np.count_nonzero(std == 0)),
                   row_order="unchanged", development=development)
    return dict(features=X, fit_features=Z, mean=mean, scale=scale,
                evaluation_indices=sample, summary=summary)


def load_config(path, *, dtype=None, development=False):
    """读取配置；命令行显式指定的精度覆盖文件设置。"""
    config = json.loads(Path(path).read_text())
    if dtype is not None:
        config["dtype"] = dtype
    return validate_config(config, development=development)
