"""保存候选、最终模型和实验记录；文件哈希用于核对输出来源。"""
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import scipy
import sklearn

from utils.data import METRIC_SPECS

MODEL_FIELDS = ("centers", "weights", "means", "variances")

def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_clusters(labels, n_samples, *, development=False):
    if labels.shape != (n_samples,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("clusters must be a one-dimensional integer array in input row order")
    if np.any(labels < 0):
        raise ValueError("every sample must be assigned a non-negative cluster ID")
    ids, sizes = np.unique(labels, return_counts=True)
    minimum = 1 if development else 5
    if not minimum <= len(ids) <= 30:
        raise ValueError(f"actual nonempty cluster count outside [{minimum},30]: {len(ids)}")
    return dict(shape=list(labels.shape), dtype=str(labels.dtype), actual_k=len(ids),
                cluster_ids=ids.tolist(), cluster_sizes=sizes.tolist(), assigned=int(sizes.sum()))


def create_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError("output directory must be empty; choose a new directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_candidate(output, row, fit, mean, scale):
    folder = output / "candidates" / row["key"]
    folder.mkdir(parents=True, exist_ok=False)
    if row["status"] == "succeeded":
        np.save(folder / "clusters.npy", fit["labels"].astype(np.int32))
        np.savez(folder / "model.npz", method=row["method"], preprocessing_mean=mean,
                 preprocessing_scale=scale, **{key: fit[key] for key in MODEL_FIELDS if key in fit})
        row["model_sha256"] = sha256(folder / "model.npz")
        row["clusters_sha256"] = sha256(folder / "clusters.npy")
    save_json(folder / "fit.json", dict(run=row, restarts=fit["restarts"]))
    save_json(output / (row["key"] + "_log.json"), fit["restarts"])


def export_selected(output, chosen, labels, *, development=False):
    source = output / "candidates" / chosen["key"]
    protocol = validate_clusters(labels, len(labels), development=development)
    shutil.copyfile(source / "clusters.npy", output / "clusters.npy")
    shutil.copyfile(source / "model.npz", output / "model.npz")
    save_json(output / "export_manifest.json", dict(run_key=chosen["key"], seed=chosen["seed"],
              restart=chosen["best_restart"], model_sha256=sha256(output / "model.npz"),
              clusters_sha256=sha256(output / "clusters.npy")))
    return protocol


def begin_experiment(output, config, dataset, *, input_path, config_path,
                     implementation_path, device_info, stage, development=False,
                     source=None, source_plan=None, fit_config=None):
    """保存本次输入与配置，返回选择规则和初始运行状态。"""
    plan = dict(source_plan) if source_plan is not None else dict(
        input_sha256=sha256(input_path), implementation_sha256=sha256(implementation_path),
        development=development, device=device_info["device"])
    # 回放保留原拟合身份，但不沿用旧版源码快照记录。
    for key in ("fit_source_hashes", "evaluation_source_hashes", "core_sha256", "run_sha256"):
        plan.pop(key, None)
    plan.update(schema_version=3, stage=stage, dtype=config["dtype"],
                source_config_sha256=sha256(config_path), teacher_labels_used=False,
                evaluation_implementation_sha256=sha256(implementation_path) if stage != "fit" else None,
                selection_rule=dict(metric=config["selection_metric"], aggregation="mean",
                    direction=METRIC_SPECS[config["selection_metric"]]["direction"],
                    group_tie_atol=1e-6, group_tie_break=["smaller_k", "kmeans_first"],
                    representative_run="best_training_objective", representative_seed_tie_rtol=1e-7))
    if source is not None:
        plan["source_plan_sha256"] = sha256(source / "plan.json")
    # 当前评价配置与原始拟合配置分别保存，回放时可核验哪些设置发生了变化。
    save_json(output / "config.json", config)
    save_json(output / "fit_config.json", fit_config if fit_config is not None else config)
    plan["config_sha256"] = sha256(output / "config.json")
    plan["fit_config_sha256"] = sha256(output / "fit_config.json")
    save_json(output / "plan.json", plan)
    save_json(output / "input_summary.json", dataset["summary"])
    np.save(output / "diagnostic_indices.npy", dataset["evaluation_indices"])
    np.savez(output / "preprocessing.npz", mean=dataset["mean"], scale=dataset["scale"])
    manifest = dict(schema_version=3, status="running", stage=stage, development=development,
                    device=device_info["device"], dtype=config["dtype"], device_info=device_info,
                    planned_runs=2 * len(config["candidate_k"]) * len(config["seeds"]))
    save_json(output / "manifest.json", manifest)
    return plan, manifest


def finish_experiment(output, config, plan, manifest, *, runs, groups, selection,
                      selected, protocol, device_info, elapsed):
    """写入汇总和完成状态；有最终选择不代表每个候选都成功。"""
    save_json(output / "runs.json", runs)
    save_json(output / "groups.json", groups)
    save_json(output / "selection.json", selection)
    report = dict(schema_version=3, plan=plan, config=config, runs=runs, groups=groups,
                  selected=selected, selection=selection, protocol=protocol, evaluation_space="raw",
                  clusters_sha256=sha256(output / "clusters.npy") if selected is not None else None,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, dtype=config["dtype"], **device_info),
                  elapsed_seconds=elapsed)
    save_json(output / "results.json", report)
    completed = len(runs) == manifest["planned_runs"]
    all_fits_ok = completed and all(run["status"] == "succeeded" for run in runs)
    all_metrics_ok = all(metric["status"] == "ok" for run in runs for metric in run.get("metrics", {}).values())
    manifest.update(completed_runs=len(runs), successful_fits=sum(run["status"] == "succeeded" for run in runs),
                    status="completed" if all_fits_ok and all_metrics_ok else "partial", selection_status=selection["status"])
    if manifest["stage"] != "fit" and selected is None:
        manifest["status"] = "no_eligible_group"
    if not completed:
        manifest["status"] = runs[-1]["status"] if runs else "failed"
    save_json(output / "manifest.json", manifest)
    print(f"{manifest['status']}: {output}", flush=True)
    if any(run["status"] == "interrupted" for run in runs):
        return 130
    return 0 if (manifest["stage"] == "fit" and all_fits_ok) or selected is not None else 1


def record_failure(output, manifest, error):
    """运行意外停止时保留错误，避免输出目录一直显示 running。"""
    manifest.update(status="framework_error", error=f"{type(error).__name__}: {error}")
    save_json(output / "manifest.json", manifest)
