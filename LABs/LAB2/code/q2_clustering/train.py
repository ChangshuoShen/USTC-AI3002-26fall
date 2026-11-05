"""课程提供：执行 KMeans / GMM 迭代，比较重启并遍历实验配置。"""
import numpy as np
import torch

from utils import runtime, results
from utils.runtime import validate_device
import time


def tensor_norm(value):
    return float(torch.linalg.vector_norm(value).item())


def kmeans_inertia(X, centers, labels):
    squared = (X - centers[labels]) ** 2
    return float(squared.sum().item())


def export_model(fitted):
    for name in ("labels", "centers", "weights", "means", "variances"):
        if name in fitted and isinstance(fitted[name], torch.Tensor):
            fitted[name] = fitted[name].detach().cpu().numpy()
    fitted["labels"] = np.asarray(fitted["labels"], dtype=np.int32)
    return fitted


class FitFailed(RuntimeError):
    """所有重启失败时，将完整轨迹交给实验入口保存。"""

    def __init__(self, method, restarts):
        super().__init__(f"all {method} initializations failed; inspect restart logs")
        self.restarts = restarts


def validate_tensor(value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.device != runtime.device
            or tuple(value.shape) != tuple(shape) or value.dtype != runtime.dtype
            or not bool(torch.isfinite(value).all())):
        precision = str(runtime.dtype).removeprefix("torch.")
        raise ValueError(f"{name}: expected finite {precision} tensor on {runtime.device} with shape {shape}")
    return value


def assign_and_validate(X, centers, implementation):
    labels, distances = implementation.assign_clusters(X, centers)
    if (not isinstance(labels, torch.Tensor) or labels.device != runtime.device
            or labels.shape != (len(X),) or labels.dtype not in (torch.int32, torch.int64)
            or bool((labels < 0).any()) or bool((labels >= len(centers)).any())):
        raise ValueError("assign_clusters: expected integer labels on the configured device")
    distances = validate_tensor(distances, (len(X),), "assign_clusters distances")
    if bool((distances < 0).any()):
        raise ValueError("assign_clusters: negative squared distances")
    return labels, distances


def estep_and_validate(X, weights, means, variances, implementation):
    resp, likelihood = implementation.gmm_estep(X, weights, means, variances)
    resp = validate_tensor(resp, (len(X), len(weights)), "gmm_estep resp")
    valid = (resp.device == runtime.device and not bool((resp < 0).any())
             and torch.allclose(resp.sum(dim=1), torch.ones(len(X), dtype=runtime.dtype,
                                                            device=runtime.device),
                                **runtime.normalization_tolerances()))
    if isinstance(likelihood, torch.Tensor):
        if likelihood.ndim != 0 or likelihood.device != runtime.device:
            raise ValueError("gmm_estep: likelihood must be scalar on the configured device")
        likelihood = likelihood.item()
    if not valid:
        raise ValueError("gmm_estep: responsibilities must be nonnegative and sum to one")
    if (not isinstance(likelihood, (float, int, np.floating, np.integer))
            or isinstance(likelihood, (bool, np.bool_)) or not np.isfinite(likelihood)):
        raise ValueError("gmm_estep: mean likelihood must be a finite scalar")
    return resp, float(likelihood)


def prepare_fit_input(X, k, n_init, max_iter, tol):
    validate_device(runtime.device.type)
    X = torch.as_tensor(X, dtype=runtime.dtype, device=runtime.device)
    if (X.ndim != 2 or X.shape[1] < 1 or not bool(torch.isfinite(X).all())
            or not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_))
            or not 1 <= k <= len(X)):
        raise ValueError("finite (N,D) data and 1 <= K <= N required")
    if (not isinstance(n_init, (int, np.integer)) or isinstance(n_init, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, (bool, np.bool_))
            or n_init < 1 or max_iter < 1 or not np.isfinite(tol) or tol < 0):
        raise ValueError("invalid restart/iteration/tolerance configuration")
    return X


def fit_kmeans(X, k, seed, implementation, n_init=3, max_iter=200, tol=1e-4):
    """执行 KMeans 迭代，保留惯性最小的成功重启。"""
    # 每个候选有自己的主种子，每次重启再从中生成子种子。
    X = prepare_fit_input(X, k, n_init, max_iter, tol)
    master = torch.Generator().manual_seed(seed)
    best, histories = None, []
    for restart in range(n_init):
        restart_seed = int(torch.randint(0, 2**32, (1,), generator=master).item())
        history, previous_labels = [], None
        converged, reason = False, "max_iter"
        try:
            centers = validate_tensor(
                implementation.init_centers_kmeanspp(X, k, torch.Generator(device=runtime.device).manual_seed(restart_seed)),
                (k, X.shape[1]), "init_centers_kmeanspp")
            for iteration in range(1, max_iter + 1):
                labels, _ = assign_and_validate(X, centers, implementation)
                updated = validate_tensor(implementation.update_centers(X, labels, centers), centers.shape,
                                       "update_centers")
                shift = tensor_norm(updated - centers)
                relative_shift = shift / max(1.0, tensor_norm(centers))
                centers = updated
                objective = kmeans_inertia(X, centers, labels)
                if not np.isfinite(objective):
                    raise ValueError("non-finite KMeans objective")
                history.append(dict(iteration=iteration, objective=objective,
                                    relative_center_shift=relative_shift,
                                    phase="after_mean_update_before_final_reassignment"))
                unchanged = (torch.equal(labels, previous_labels)
                             if previous_labels is not None else False)
                if unchanged:
                    converged, reason = True, "labels_unchanged"
                    break
                if relative_shift <= tol:
                    converged, reason = True, "relative_center_shift"
                    break
                previous_labels = labels.clone()
            # 使用最终中心重新分配，保证模型参数与导出的簇编号一致。
            labels, _ = assign_and_validate(X, centers, implementation)
            inertia = kmeans_inertia(X, centers, labels)
            if not np.isfinite(inertia):
                raise ValueError("non-finite final KMeans objective")
            record = dict(restart=restart, restart_seed=restart_seed, n_iter=iteration,
                          converged=converged, stop_reason=reason, objective=inertia,
                          history=history)
            histories.append(record)
            if best is None or inertia < best["objective"]:
                best = dict(record, labels=labels.clone(), centers=centers.clone())
        except (ValueError, FloatingPointError) as error:
            histories.append(dict(restart=restart, restart_seed=restart_seed,
                                  converged=False, stop_reason="failed", error=str(error),
                                  history=history))
    if best is None:
        raise FitFailed("KMeans", histories)
    best["restarts"] = histories
    return export_model(best)


def fit_gmm(X, k, seed, implementation, n_init=3, max_iter=100, tol=1e-4, reg_covar=1e-4):
    """执行对角 GMM 的 EM 迭代，保留平均对数似然最大的成功重启。"""
    # 每个候选有自己的主种子，每次重启再从中生成子种子。
    X = prepare_fit_input(X, k, n_init, max_iter, tol)
    if not np.isfinite(reg_covar) or reg_covar <= 0:
        raise ValueError("reg_covar must be positive")
    master = torch.Generator().manual_seed(seed)
    best, histories = None, []
    for restart in range(n_init):
        restart_seed = int(torch.randint(0, 2**32, (1,), generator=master).item())
        weights = torch.full((k,), 1.0 / k, dtype=runtime.dtype, device=runtime.device)
        variances = (X.var(dim=0, correction=0) + reg_covar).repeat(k, 1)
        history, decreases = [], 0
        converged, reason = False, "max_iter"
        try:
            means = validate_tensor(
                implementation.init_centers_kmeanspp(X, k, torch.Generator(device=runtime.device).manual_seed(restart_seed)),
                (k, X.shape[1]), "init_centers_kmeanspp")
            resp, previous = estep_and_validate(X, weights, means, variances, implementation)
            history.append(dict(iteration=0, mean_log_likelihood=float(previous)))
            for iteration in range(1, max_iter + 1):
                weights, means, variances = implementation.gmm_mstep(X, resp, reg_covar)
                weights = validate_tensor(weights, (k,), "gmm_mstep weights")
                means = validate_tensor(means, (k, X.shape[1]), "gmm_mstep means")
                variances = validate_tensor(variances, means.shape, "gmm_mstep variances")
                valid = (not bool((weights <= 0).any())
                         and bool(torch.isclose(weights.sum(), weights.new_tensor(1.0),
                                                 **runtime.normalization_tolerances()))
                         and not bool((variances <= 0).any()))
                if not valid:
                    raise ValueError("gmm_mstep: weights must be positive and normalized; variances must be positive")
                # M 步更新后重新计算似然；小幅下降也不能当作收敛。
                resp, current = estep_and_validate(X, weights, means, variances, implementation)
                delta = float(current - previous)
                history.append(dict(iteration=iteration, mean_log_likelihood=float(current),
                                    change=delta, phase="after_parameter_update"))
                decreases += int(delta < 0)
                if 0 <= delta <= tol:
                    converged, reason = True, "nonnegative_mean_likelihood_change"
                    break
                previous = current
            record = dict(restart=restart, restart_seed=restart_seed, n_iter=iteration,
                          converged=converged, stop_reason=reason, objective=float(current),
                          likelihood_decreases=decreases, history=history)
            histories.append(record)
            if best is None or current > best["objective"]:
                best = dict(record, labels=resp.argmax(dim=1), weights=weights.clone(),
                            means=means.clone(), variances=variances.clone())
        except (ValueError, FloatingPointError) as error:
            histories.append(dict(restart=restart, restart_seed=restart_seed,
                                  converged=False, stop_reason="failed", error=str(error),
                                  history=history))
    if best is None:
        raise FitFailed("GMM", histories)
    best["restarts"] = histories
    return export_model(best)


def predict(X, fitted, method, implementation):
    validate_device(runtime.device.type)
    X = torch.as_tensor(X, dtype=runtime.dtype, device=runtime.device)
    parameters = {name: torch.as_tensor(fitted[name], dtype=runtime.dtype, device=runtime.device)
                  for name in ("centers", "weights", "means", "variances") if name in fitted}
    if method == "kmeans":
        labels = assign_and_validate(X, parameters["centers"], implementation)[0]
    elif method == "gmm":
        resp, _ = estep_and_validate(X, parameters["weights"], parameters["means"], parameters["variances"], implementation)
        labels = resp.argmax(dim=1)
    else:
        raise ValueError("method must be kmeans or gmm")
    return labels.detach().cpu().numpy().astype(np.int32)


def fit_candidates(dataset, config, implementation, output):
    """按方法、K 和 seed 拟合；每个候选保存自己的参数、分区和重启轨迹。"""
    runs, partitions = [], {}
    for method in ("kmeans", "gmm"):
        for k in config["candidate_k"]:
            for seed in config["seeds"]:
                key = f"{method}_k{k}_seed{seed}"
                row = dict(key=key, method=method, k=k, seed=seed)
                started = time.perf_counter()
                stop = False
                try:
                    if method == "kmeans":
                        fit = fit_kmeans(dataset["fit_features"], k, seed, implementation, config["n_init"],
                                         config["kmeans_max_iter"], config["kmeans_tol"])
                    else:
                        fit = fit_gmm(dataset["fit_features"], k, seed, implementation, config["n_init"],
                                      config["gmm_max_iter"], config["gmm_tol"], config["reg_covar"])
                    labels = fit["labels"]
                    ids, sizes = np.unique(labels, return_counts=True)
                    row.update(status="succeeded", actual_k=len(ids), cluster_ids=ids.tolist(),
                               cluster_sizes=sizes.tolist(), best_restart=fit["restart"],
                               objective=float(fit["objective"]),
                               objective_name="inertia" if method == "kmeans" else "mean_log_likelihood",
                               objective_space="standardized" if config["standardize"] else "raw",
                               n_iter=fit["n_iter"], converged=fit["converged"], stop_reason=fit["stop_reason"])
                    partitions[key] = labels
                except FitFailed as exc:
                    fit = dict(restarts=exc.restarts)
                    row.update(status="failed", error=str(exc), stop_reason="all_restarts_failed")
                except NotImplementedError as exc:
                    fit = dict(restarts=[])
                    row.update(status="not_implemented", error=str(exc), stop_reason="not_implemented")
                    stop = True
                except KeyboardInterrupt:
                    fit = dict(restarts=[])
                    row.update(status="interrupted", error="user interrupted fit", stop_reason="interrupted")
                    stop = True
                # 失败候选也落盘；TODO 未补完或用户中断时停止后续拟合。
                row["elapsed_seconds"] = time.perf_counter() - started
                results.save_candidate(output, row, fit, dataset["mean"], dataset["scale"])
                runs.append(row)
                results.save_json(output / "runs.json", runs)
                print(f"{key}: {row['status']}; stop={row['stop_reason']}", flush=True)
                if stop:
                    return runs, partitions
    return runs, partitions
