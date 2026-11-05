"""额外功能：分阶段运行、评价回放、开发模式、自检、输出核验和标签归档。"""
import argparse
import importlib.util
import json
import shutil
import time
from functools import partial, wraps
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from utils import runtime, results
from utils.data import EVALUATION_FIELDS, load_features, prepare_data, validate_config
from utils.results import save_json, sha256, validate_clusters
from train import fit_kmeans, fit_gmm, predict
from evaluate import evaluate_candidates

PACKAGE = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE / "data/features.npz"
DEFAULT_OUTPUT = PACKAGE / "outputs_final"
DEFAULT_IMPLEMENTATION = PACKAGE / "submission.py"


# 实现替换与归档：常规入口也复用这两个小工具。
def load_implementation(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("student_kernels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_candidate_archive(output, partitions):
    """按运行名压缩保存所有成功候选的簇编号。"""
    np.savez_compressed(output / "candidate_clusters.npz", **partitions)


# 评价回放：先验证历史配置与文件，再读取已有候选。
def validate_saved_experiment(directory, plan):
    """验证保存的配置，返回当前评价配置和最初拟合配置；不保存或读取源码。"""
    directory = Path(directory)

    def saved_file(relative):
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"missing saved experiment file: {relative}; restore the original experiment or refit")
        return path

    def verify(relative, digest):
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"missing saved hash for {relative}; restore the original experiment or refit")
        path = saved_file(relative)
        if sha256(path) != digest:
            raise ValueError(f"saved {relative}: hash mismatch; restore the original experiment or refit. "
                             "Use a separate --config file to change evaluation settings.")
        return path

    # 必须先验证历史文件，再读取它作为回放参数的比较基准，避免与被修改后的自身比较。
    cfg_path = verify("config.json", plan.get("config_sha256"))
    fit_relative = "fit_config.json"
    # 仅兼容旧结果的配置文件位置；旧目录里的源码不读取，也不复制到新结果。
    if not (directory / fit_relative).is_file() and plan.get("schema_version", 2) < 3:
        fit_relative = "source_snapshot/fit_config.json"
    fit_path = (verify(fit_relative, plan.get("fit_config_sha256"))
                if "fit_config_sha256" in plan or plan.get("schema_version", 2) >= 3
                else saved_file(fit_relative))
    configs = [json.loads(path.read_text()) for path in (cfg_path, fit_path)]
    for cfg in configs:
        cfg.setdefault("dtype", "float64")  # 未记录精度的旧实验使用 float64。
    saved, fitted = [validate_config(cfg, development=plan["development"]) for cfg in configs]
    changed = sorted(key for key in saved.keys() | fitted.keys()
                     if key not in EVALUATION_FIELDS and saved.get(key) != fitted.get(key))
    if changed:
        raise ValueError(f"saved fitting configuration does not match fit_config.json: {changed}; refit required")
    return saved, fitted


def copy_saved_runs(source, output):
    rows = json.loads((source / "runs.json").read_text())
    labels_by_run = {}
    for row in rows:
        folder = source / "candidates" / row["key"]
        if row["status"] == "succeeded":
            for name, digest in (("model.npz", row["model_sha256"]), ("clusters.npy", row["clusters_sha256"])):
                if sha256(folder / name) != digest:
                    raise ValueError(f"saved candidate hash mismatch: {row['key']}/{name}")
            labels_by_run[row["key"]] = np.load(folder / "clusters.npy", allow_pickle=False)
        shutil.copytree(folder, output / "candidates" / row["key"])
        restarts = json.loads((folder / "fit.json").read_text())["restarts"]
        save_json(output / (row["key"] + "_log.json"), restarts)
    return rows, labels_by_run


def evaluate_saved(args):
    """用当前评价实现重新评价已有候选，不调用拟合函数。"""
    source = args.experiment.resolve()
    source_plan = json.loads((source / "plan.json").read_text())
    saved_config, fit_config = validate_saved_experiment(source, source_plan)
    device = runtime.resolve_device(args.device, source_plan.get("device"))
    if source_plan.get("device", "cpu") != device:
        raise ValueError("evaluation replay must use the device recorded by the fit stage")
    config_path = args.config or source / "config.json"
    raw = json.loads(config_path.read_text())
    raw.setdefault("dtype", saved_config["dtype"])
    if args.dtype is not None:
        raw["dtype"] = args.dtype
    config = validate_config(raw, development=args.development)
    if any(config[key] != saved_config[key] for key in config if key not in EVALUATION_FIELDS):
        raise ValueError("evaluation replay cannot change fitting configuration")
    if source_plan["development"] != args.development or sha256(args.input) != source_plan["input_sha256"]:
        raise ValueError("evaluation replay requires the original input and development profile")
    device_info = runtime.configure(device, config["dtype"], config["threads"])
    dataset = prepare_data(args.input, config, development=args.development)
    implementation = load_implementation(args.implementation)
    output = results.create_output(args.output)
    plan, manifest = results.begin_experiment(output, config, dataset, input_path=args.input,
        config_path=config_path, implementation_path=args.implementation, device_info=device_info,
        stage="evaluate", development=args.development, source=source, source_plan=source_plan, fit_config=fit_config)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=config["threads"]):
            runs, partitions = copy_saved_runs(source, output)
            groups, selection, selected = evaluate_candidates(dataset, runs, partitions, config,
                implementation, output, plan["selection_rule"], development=args.development)
            protocol = (results.export_selected(output, selected, partitions[selected["key"]], development=args.development)
                        if selected is not None else None)
            save_candidate_archive(output, partitions)
        return results.finish_experiment(output, config, plan, manifest, runs=runs, groups=groups,
            selection=selection, selected=selected, protocol=protocol, device_info=device_info,
            elapsed=time.perf_counter() - started)
    except Exception as error:
        results.record_failure(output, manifest, error)
        raise


# 公开自检：仅使用小型合成数据，不读取正式特征或真实标签。
def torch_check_adapter(implementation):
    """Run public NumPy fixtures through the unified PyTorch student API."""
    selected_device = runtime.device
    selected_dtype = runtime.dtype

    def wrap(function):
        @wraps(function)
        def checked(*arguments):
            fixture = bool(arguments) and isinstance(arguments[0], np.ndarray)
            converted = tuple(
                torch.as_tensor(
                    value,
                    device=selected_device,
                    dtype=(torch.int64 if np.issubdtype(value.dtype, np.integer)
                           else selected_dtype),
                ) if isinstance(value, np.ndarray) else value
                for value in arguments
            )
            originals = [(value, value.clone()) for value in converted
                         if fixture and isinstance(value, torch.Tensor)]
            with torch.no_grad():
                result = function(*converted)
            if any(not torch.equal(value, original) for value, original in originals):
                raise ValueError("student hook modified an input tensor")
            values = result if isinstance(result, tuple) else (result,)
            for value in values:
                if isinstance(value, (np.ndarray, list)):
                    raise ValueError("student hook must return PyTorch tensors or scalars")
                if isinstance(value, torch.Tensor) and value.device != selected_device:
                    raise ValueError("student hook returned a tensor on the wrong device")
            if not fixture:
                return result
            converted_result = tuple(value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                                     else value for value in values)
            return converted_result if isinstance(result, tuple) else converted_result[0]
        return checked

    names = ("init_centers_kmeanspp", "assign_clusters", "update_centers", "gmm_estep", "gmm_mstep",
             "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score")
    return SimpleNamespace(**{name: wrap(getattr(implementation, name))
                              for name in names if hasattr(implementation, name)})


def check_algorithms(implementation):
    tolerances = runtime.check_tolerances()
    assert_close = partial(np.testing.assert_allclose, **tolerances)
    completed = []
    X = np.array([[0.], [1.], [3.], [10.]])
    generator = torch.Generator(device=runtime.device).manual_seed(17)
    centers = implementation.init_centers_kmeanspp(X, 4, generator)
    np.testing.assert_array_equal(np.sort(centers[:, 0]), X[:, 0])
    generator.manual_seed(17)
    np.testing.assert_array_equal(centers, implementation.init_centers_kmeanspp(X, 4, generator))
    second_given_zero = []
    for _ in range(512):
        drawn = implementation.init_centers_kmeanspp(X, 2, generator)
        if drawn[0, 0] == 0:
            second_given_zero.append(drawn[1, 0])
    assert len(second_given_zero) >= 60, "first center should be sampled uniformly"
    assert np.mean(np.array(second_given_zero) == 10) > 0.75, "second center must use squared-distance weights"
    identical = implementation.init_centers_kmeanspp(np.zeros((5, 2)), 3, generator)
    assert identical.shape == (3, 2) and np.isfinite(identical).all()
    completed.append("kmeans++ weighted initialization and zero-distance fallback")

    labels, distances = implementation.assign_clusters(np.array([[0.], [1.], [3.]]), np.array([[0.], [2.]]))
    np.testing.assert_array_equal(labels, [0, 0, 1])
    assert_close(distances, [0, 1, 1])
    completed.append("nearest-center assignment including a tie")

    old = np.array([[8.], [9.], [30.]])
    updated = implementation.update_centers(np.array([[0.], [2.], [10.]]), np.array([0, 0, 1]), old)
    assert_close(updated, [[1.], [10.], [30.]])
    np.testing.assert_array_equal(old, [[8.], [9.], [30.]])
    completed.append("mean update, empty center and no input mutation")

    resp, ll = implementation.gmm_estep(np.array([[0.], [2.]]), np.array([.5, .5]), np.array([[0.], [2.]]), np.ones((2, 1)))
    a = 1/(1+np.exp(-2))
    assert_close(resp, [[a, 1-a], [1-a, a]])
    expected_ll = np.log1p(np.exp(-2)) - np.log(2) - .5*np.log(2*np.pi)
    assert_close(ll, expected_ll)
    high = np.full((3, 512), 100.)
    high_resp, high_ll = implementation.gmm_estep(high, np.array([.5, .5]), np.stack([np.zeros(512), np.ones(512)]), np.ones((2, 512)))
    assert np.isfinite(high_ll) and np.isfinite(high_resp).all()
    np.testing.assert_allclose(high_resp.sum(axis=1), 1, **runtime.normalization_tolerances())
    completed.append("E-step hand calculation and high-dimensional underflow handling")

    weights, means, variances = implementation.gmm_mstep(np.array([[0.], [2.], [4.]]), np.array([[1., 0.], [.5, .5], [0., 1.]]), 1e-4)
    assert_close(weights, [.5, .5])
    assert_close(means[:, 0], [2/3, 10/3])
    assert_close(variances[:, 0], [8/9+1e-4]*2)
    try:
        implementation.gmm_mstep(np.array([[0.], [2.]]), np.array([[1., 0.], [1., 0.]]), 1e-4)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-effective-count component must be reported")
    completed.append("M-step weighted moments, regularization and degenerate component")

    rng = np.random.default_rng(812)
    toy = np.concatenate([rng.normal(-4, .3, (60, 2)), rng.normal(4, .3, (60, 2))])
    for method, fitter in [("kmeans", fit_kmeans), ("gmm", fit_gmm)]:
        model = fitter(toy, 2, 17, implementation, n_init=2, max_iter=50, tol=1e-6)
        out = predict(toy, model, method, implementation)
        assert len(set(out[:60])) == len(set(out[60:])) == 1 and out[0] != out[-1]
        assert model["converged"]
        again = fitter(toy, 2, 17, implementation, n_init=2, max_iter=50, tol=1e-6)
        np.testing.assert_array_equal(out, predict(toy, again, method, implementation))
        for restart in model["restarts"]:
            field = "objective" if method == "kmeans" else "mean_log_likelihood"
            values = np.array([step[field] for step in restart["history"]])
            differences = np.diff(values)
            roundoff = tolerances["atol"] + tolerances["rtol"] * np.abs(values[:-1])
            assert np.all(differences <= roundoff) if method == "kmeans" else np.all(differences >= -roundoff)
        completed.append(method + " complete fit, trajectory and reproducibility on synthetic data")
    return completed


def check_metrics(implementation):
    assert_close = partial(np.testing.assert_allclose, **runtime.check_tolerances())
    X = np.array([[0.], [2.], [10.], [12.]])
    labels = np.array([7, 7, 30, 30])
    functions = [implementation.silhouette_score, implementation.calinski_harabasz_score,
                 implementation.davies_bouldin_score]
    expected = [np.mean([9/11, 7/9, 7/9, 9/11]), 50., .2]
    completed = []
    for function, value in zip(functions, expected):
        before, labels_before = X.copy(), labels.copy()
        assert_close(function(X, labels), value)
        np.testing.assert_array_equal(X, before)
        np.testing.assert_array_equal(labels, labels_before)
        assert_close(function(X, np.array([1, 1, 0, 0])), value)
        completed.append(function.__name__ + " hand calculation, label invariance and no mutation")
    # Singleton silhouette is zero; distances are 1, 9 and 10, not squared distances.
    assert_close(implementation.silhouette_score(np.array([[0.], [1.], [10.]]),
                                                np.array([0, 0, 1])), (0.9 + 8/9)/3)
    completed.append("silhouette singleton convention")
    # Each cluster has scatter `spread`; their centers are `gap` apart.
    # Orthogonal coordinates keep both quantities independent at tiny scales.
    for spread, gap, value in [(0.5e-8, 8e-8, 0.), (1e-8, 8e-8, 0.),
                               (2e-8, 8e-8, .5), (1e-4, 0.5e-8, 0.),
                               (1e-4, 1e-8, 0.), (1e-4, 2e-8, 1e4)]:
        points = np.array([[-spread, 0.], [spread, 0.],
                           [-spread, gap], [spread, gap]])
        assert_close(implementation.davies_bouldin_score(points, labels), value)
    # A third, distant center prevents the all-centers-near-zero convention.
    # Only exact zero distances are excluded; tiny positive gaps still count.
    for gap in (0., 5e-9):
        points = np.array([[-1., 0.], [1., 0.], [-2., gap], [2., gap],
                           [-1., 10.], [1., 10.]])
        value = 4/15 if gap == 0 else (6/gap + 3/(10-gap))/3
        assert_close(implementation.davies_bouldin_score(points, np.array([0, 0, 1, 1, 2, 2])),
                     value)
    completed.append("DB absolute zero tolerance and coincident centers")
    return completed


def self_check(args):
    runtime.configure(args.device, args.dtype or "float32", 4)
    implementation = torch_check_adapter(load_implementation(args.implementation))
    completed = []
    with threadpool_limits(limits=4):
        if args.part in ("all", "algorithms"):
            completed.extend(check_algorithms(implementation))
        if args.part in ("all", "metrics"):
            completed.extend(check_metrics(implementation))
    for name in completed:
        print("PASS:", name)
    print(f"{len(completed)} checks passed; also inspect your real-data results and logs.")
    return 0


# 输出核验：只检查格式、哈希与来源，不重新预测或计算训练目标。
def audit_provenance(output, result):
    """核对最终选择、运行记录、候选与导出文件；来源材料缺失也视为失败。"""
    def required_file(relative):
        path = output / relative
        if not path.is_file():
            raise ValueError(f"missing required output artifact: {relative}")
        return path

    selection = json.loads(required_file("selection.json").read_text())
    export = json.loads(required_file("export_manifest.json").read_text())
    runs = json.loads(required_file("runs.json").read_text())
    chosen = result.get("selected")
    fields = ("key", "method", "k", "seed", "status", "best_restart", "objective",
              "model_sha256", "clusters_sha256")
    if not isinstance(chosen, dict) or any(field not in chosen for field in fields):
        raise ValueError("results.json is missing the selected run record")
    key = chosen["key"]
    if not isinstance(key, str) or key in ("", ".", "..") or Path(key).name != key:
        raise ValueError("invalid selected run key")

    # selection.json 与 results.json 各自保存一份选择，二者都必须指向同一运行。
    expected_selection = dict(status="selected", run_key=key, method=chosen["method"],
                              k=chosen["k"], restart=chosen["best_restart"])
    if (not isinstance(selection, dict) or selection != result.get("selection")
            or any(selection.get(name) != value for name, value in expected_selection.items())):
        raise ValueError("selection.json does not match the selected run in results.json")
    if not isinstance(runs, list) or runs != result.get("runs"):
        raise ValueError("runs.json does not match the run records in results.json")
    matches = [row for row in runs if row.get("key") == key]
    if len(matches) != 1 or matches[0] != chosen or chosen["status"] != "succeeded":
        raise ValueError("selected run must have one matching successful record in runs.json")

    candidate = Path("candidates") / key
    fit = json.loads(required_file(candidate / "fit.json").read_text())
    fit_run = fit.get("run", {})
    # fit.json 在评价前保存，因此只比较拟合字段，允许汇总记录多出评价指标。
    if (any(field not in fit_run for field in fields)
            or any(chosen.get(name) != value for name, value in fit_run.items())):
        raise ValueError("selected candidate fit.json does not match the selected run")
    expected_export = dict(run_key=key, seed=chosen["seed"], restart=chosen["best_restart"])
    if not isinstance(export, dict) or any(export.get(name) != value for name, value in expected_export.items()):
        raise ValueError("export_manifest.json does not identify the selected run")

    # 文件必须存在且内容一致；不能因为清单缺失就跳过来源检查。
    for name in ("model.npz", "clusters.npy"):
        digest = sha256(required_file(name))
        hash_field = name.split(".")[0] + "_sha256"
        if sha256(required_file(candidate / name)) != digest or chosen[hash_field] != digest:
            raise ValueError(f"selected candidate/export file mismatch: {name}")
        if export.get(hash_field) != digest:
            raise ValueError(f"export_manifest.json hash mismatch: {name}")


def check_output(args):
    X = load_features(args.input, development=args.development)
    labels = np.load(args.output / "clusters.npy", allow_pickle=False)
    report = validate_clusters(labels, len(X), development=args.development)
    result = json.loads((args.output / "results.json").read_text())
    plan = json.loads((args.output / "plan.json").read_text())
    if plan != result["plan"]:
        raise ValueError("plan.json does not match results.json")
    config, _ = validate_saved_experiment(args.output, plan)
    recorded_config = dict(result["config"])
    recorded_config.setdefault("dtype", "float64")
    if config != recorded_config:
        raise ValueError("config.json does not match results.json")
    if sha256(args.input) != result["plan"]["input_sha256"] or sha256(args.output / "clusters.npy") != result["clusters_sha256"]:
        raise ValueError("input/output hash mismatch")
    audit_provenance(args.output, result)
    report.update(passed=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行开发数据或只拟合候选")
    run.add_argument("--stage", choices=("all", "fit"), default="all")
    run.add_argument("--config", type=Path, default=PACKAGE / "config.json")
    replay = commands.add_parser("evaluate", help="重新评价已保存的候选")
    replay.add_argument("--experiment", type=Path, required=True)
    replay.add_argument("--config", type=Path)
    check = commands.add_parser("check-output", help="核对输出文件及其来源")
    selfcheck = commands.add_parser("self-check", help="运行算法和评价函数的小例子")
    selfcheck.add_argument("--part", choices=("all", "algorithms", "metrics"), default="all")
    for command in (run, replay, check):
        command.add_argument("--input", type=Path, default=DEFAULT_INPUT)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--development", action="store_true", help="使用小型合成数据，仅用于调试")
    for command in (run, replay, selfcheck):
        command.add_argument("--device", choices=("cpu", "cuda"))
        command.add_argument("--dtype", choices=("float32", "float64"))
        command.add_argument("--implementation", type=Path, default=DEFAULT_IMPLEMENTATION)
    args = parser.parse_args()
    try:
        if args.command == "self-check":
            code = self_check(args)
        elif args.command == "check-output":
            print(json.dumps(check_output(args), ensure_ascii=False, indent=2))
            code = 0
        elif args.command == "evaluate":
            code = evaluate_saved(args)
        else:
            # 延迟导入，避免常规入口与辅助入口相互导入时重复初始化。
            from main import run
            code = run(args.config, args.output, args.device, input_path=args.input,
                       implementation_path=args.implementation, dtype=args.dtype,
                       development=args.development, fit_only=args.stage == "fit")
    except NotImplementedError as error:
        parser.exit(1, f"{error}\n")
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
