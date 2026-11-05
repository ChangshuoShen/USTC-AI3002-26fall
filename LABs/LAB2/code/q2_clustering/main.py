"""课程提供：读取配置、拟合候选、评价与选择，保存最终聚类结果。"""
import argparse
import time
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

import extras
from train import fit_candidates
from evaluate import evaluate_candidates
from utils import data, results, runtime

PACKAGE = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE / "config.json"
DEFAULT_INPUT = PACKAGE / "data/features.npz"
DEFAULT_OUTPUT = PACKAGE / "outputs_final"
DEFAULT_IMPLEMENTATION = PACKAGE / "submission.py"


def run(config_path=DEFAULT_CONFIG, output=DEFAULT_OUTPUT, device=None, *,
        input_path=DEFAULT_INPUT, implementation_path=DEFAULT_IMPLEMENTATION,
        dtype=None, development=False, fit_only=False):
    # 读取配置、设置设备，再准备数据；拟合特征与原始评价特征分别保留。
    config = data.load_config(config_path, dtype=dtype, development=development)
    device_info = runtime.configure(device, config["dtype"], config["threads"])
    dataset = data.prepare_data(input_path, config, development=development)
    dataset["fit_features"] = torch.as_tensor(dataset["fit_features"], dtype=runtime.dtype, device=runtime.device)
    implementation = extras.load_implementation(implementation_path)

    # 在实验开始前保存输入与配置，后续每个候选都会留下独立运行记录。
    output = results.create_output(output)
    plan, manifest = results.begin_experiment(output, config, dataset, input_path=input_path,
        config_path=config_path, implementation_path=implementation_path, device_info=device_info,
        stage="fit" if fit_only else "all", development=development)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=config["threads"]):
            # 拟合：两种方法使用同一组 K 和 seed，每个候选比较多次初始化。
            runs, partitions = fit_candidates(dataset, config, implementation, output)
            groups, selected, protocol = [], None, None
            selection = dict(status="not_requested", rule=plan["selection_rule"])
            if not fit_only:
                # 评价与选择：先比较跨种子指标，再按训练目标选代表运行。
                groups, selection, selected = evaluate_candidates(dataset, runs, partitions, config,
                    implementation, output, plan["selection_rule"], development=development)
                if selected is not None:
                    protocol = results.export_selected(output, selected, partitions[selected["key"]],
                                                       development=development)
            extras.save_candidate_archive(output, partitions)
        # 保存汇总并给出完成状态；没有有效选择时保留候选供排查。
        return results.finish_experiment(output, config, plan, manifest, runs=runs, groups=groups,
            selection=selection, selected=selected, protocol=protocol, device_info=device_info,
            elapsed=time.perf_counter() - started)
    except Exception as error:
        results.record_failure(output, manifest, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    try:
        code = run(args.config, args.output, args.device)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
