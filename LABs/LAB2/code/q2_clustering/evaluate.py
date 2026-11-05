"""调用评价函数，在统一特征空间比较候选并选择最终运行。"""
import numpy as np
import torch

from utils import runtime, results
from utils.data import METRICS, METRIC_SPECS

def compute_metric(X, labels, function):
    cluster_count = int(torch.unique(labels).numel())
    if not 2 <= cluster_count < len(X):
        return dict(value=None, status="undefined", reason="requires 2 <= actual clusters < samples")
    try:
        raw = function(X, labels)
        if isinstance(raw, torch.Tensor):
            if raw.ndim != 0 or raw.device != runtime.device:
                raise ValueError("metric must return a scalar on the configured device")
            raw = raw.item()
        if not isinstance(raw, (float, int, np.floating, np.integer)) or isinstance(raw, (bool, np.bool_)):
            raise ValueError("metric must return a scalar number")
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError("metric returned a non-finite value")
        return dict(value=value, status="ok", reason=None)
    except NotImplementedError as exc:
        return dict(value=None, status="not_implemented", reason=str(exc))
    except (ValueError, FloatingPointError, TypeError, AttributeError) as exc:
        return dict(value=None, status="failed", reason=f"{type(exc).__name__}: {exc}")


def evaluate_partition(X, labels, sample, implementation):
    X = torch.as_tensor(X, dtype=runtime.dtype, device=runtime.device)
    labels = np.asarray(labels)
    if labels.shape != (len(X),) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("evaluation requires one nonnegative integer label per input row")
    ids, sizes = np.unique(labels, return_counts=True)
    device_labels = torch.as_tensor(labels, dtype=torch.int64, device=runtime.device)
    subset = torch.as_tensor(sample, dtype=torch.int64, device=runtime.device)
    metrics = {}
    # Silhouette 取公共子集，CH 和 DB 使用全量原始特征。
    for name, spec in METRIC_SPECS.items():
        data, partition = ((X[subset], device_labels[subset]) if spec["scope"] == "subset"
                           else (X, device_labels))
        function = getattr(implementation, spec["function"], None)
        if function is None:
            metrics[name] = dict(value=None, status="failed", reason=f"missing {spec['function']}")
        else:
            metrics[name] = compute_metric(data, partition, function)
            value = metrics[name]["value"]
            low, high = spec["bounds"]
            if value is not None and ((low is not None and value < low - 1e-7)
                                      or (high is not None and value > high + 1e-7)):
                metrics[name] = dict(value=None, status="failed", reason="metric outside its valid range")
    sample_covered_clusters = int(torch.unique(device_labels[subset]).numel())
    return dict(actual_k=len(ids), cluster_ids=ids.tolist(), cluster_sizes=sizes.tolist(),
                sample_covered_clusters=sample_covered_clusters, metrics=metrics,
                **{name: metrics[name]["value"] for name in METRIC_SPECS})


def summarize_groups(rows, ks, seeds, metric="silhouette", *, development=False):
    groups = []
    for method in ("kmeans", "gmm"):
        for k in ks:
            members = [r for r in rows if r["method"] == method and r["k"] == k]
            successful = [r for r in members if r["status"] == "succeeded"]
            missing = [s for s in seeds if not any(r["seed"] == s for r in members)]
            failures = [r["seed"] for r in members if r["status"] != "succeeded"]
            # 不能丢掉失败种子后，只平均成功结果。
            reasons = []
            if missing or failures or len(successful) != len(seeds):
                reasons.append("all planned seeds must have successful fits")
            if any(not (1 if development else 5) <= r["actual_k"] <= 30 for r in successful):
                reasons.append("actual cluster count outside final protocol")
            if any(r.get(metric) is None for r in successful):
                reasons.append(f"{metric} unavailable for one or more seeds")
            group = dict(method=method, k=k, eligible=not reasons, exclusion_reasons=reasons,
                         n_planned=len(seeds), n_successful=len(successful), missing_seeds=missing,
                         failed_seeds=failures, run_keys=[r["key"] for r in members])
            for name in METRICS:
                values = [r.get(name) for r in successful]
                complete = len(values) == len(seeds) and all(v is not None for v in values)
                group[name + "_mean"] = float(np.mean(values)) if complete else None
                group[name + "_std"] = float(np.std(values, ddof=1)) if complete and len(values) > 1 else None
            groups.append(group)
    return groups


def select_group(groups, metric="silhouette"):
    if metric not in METRICS:
        raise ValueError("unknown selection metric")
    valid = [g for g in groups if g["eligible"] and g.get(metric + "_mean") is not None]
    if not valid:
        raise ValueError(f"no eligible group for {metric}")
    minimize = METRIC_SPECS[metric]["direction"] == "minimize"
    best = (min if minimize else max)(g[metric + "_mean"] for g in valid)
    tied = [g for g in valid if abs(g[metric + "_mean"] - best) <= 1e-6]
    return min(tied, key=lambda g: (g["k"], g["method"] != "kmeans"))


def choose_representative_run(rows, selected):
    choices = [r for r in rows if r["status"] == "succeeded"
               and r["method"] == selected["method"] and r["k"] == selected["k"]]
    optimum = (min if selected["method"] == "kmeans" else max)(r["objective"] for r in choices)
    tied = [r for r in choices if abs(r["objective"] - optimum) <= 1e-7 * max(1, abs(optimum))]
    return min(tied, key=lambda r: r["seed"])


def evaluate_candidates(dataset, runs, partitions, config, implementation, output, rule, *, development=False):
    """先评价每次运行，再按跨种子均值选组，最后按训练目标选代表种子。"""
    # 评价始终使用原始特征；标准化只影响拟合，不影响指标的比较空间。
    features = torch.as_tensor(dataset["features"], dtype=runtime.dtype, device=runtime.device)
    for run in runs:
        if run["status"] == "succeeded":
            run.update(evaluate_partition(features, partitions[run["key"]],
                                          dataset["evaluation_indices"], implementation))
        results.save_json(output / "runs.json", runs)
    groups = summarize_groups(runs, config["candidate_k"], config["seeds"],
                              config["selection_metric"], development=development)
    selection = dict(status="no_eligible_group", rule=rule)
    try:
        group = select_group(groups, config["selection_metric"])
    except ValueError as error:
        selection["reason"] = str(error)
        return groups, selection, None
    selected = choose_representative_run(runs, group)
    selection.update(status="selected", method=group["method"], k=group["k"],
                     run_key=selected["key"], restart=selected["best_restart"])
    return groups, selection, selected
