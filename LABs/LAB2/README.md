# Lab2：神经网络与无监督聚类

最终只需提交覆盖两题的 `report.pdf`、第一题的 `models.py` 和 `train.py`、第二题的 `submission.py` 及最终 `clusters.npy`。报告格式自定，提交目录见[实验文档](lab2.pdf)的“提交”一节。其他原始输出无需上传，请在本地保留至验收结束。

<a id="environment"></a>

## 环境搭建

两题共用一个 Python 环境，代码、数据和输出分别存放。建议 Python 3.12；以下安装命令均从本 README 所在目录执行。

### 1. 创建环境

本地可使用 venv：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

使用 Conda 时，以以下命令替代 venv 步骤；已有 `ai26` 环境可直接激活：

```bash
conda create -n ai26 python=3.12 -y
conda activate ai26
```

### 2. 安装依赖

按实际使用的设备选择一套依赖。CPU：

```bash
pip install -r code/requirements.txt
```

NVIDIA CUDA：

```bash
pip install -r code/requirements-gpu.txt
```

CUDA 依赖固定为 `torch==2.11.0+cu128`，会替换环境中已有的 PyTorch。107 平台的安装与 GPU 检查请放在计算作业内执行，见 [107 平台配置](#platform-107)。

### 3. 检查环境与设备

```bash
python -c 'import sys, torch; print("Python:", sys.version); print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

两题都可使用 CPU。使用 CUDA 时，须在实际运行实验的节点上确认检查结果为 `True`，并显式传入 `--device cuda`。

- 第一题支持 `auto/cpu/cuda/mps/xpu`，默认 `auto`；下面的命令统一指定 CPU。
- 第二题支持 `cpu/cuda`，两种设备使用同一份 PyTorch Tensor 实现；指定 CUDA 但不可用时会报错，不会自动退回 CPU。

不同软件版本或设备上的计算结果可能存在数值差异。

<a id="q1"></a>

## 第一题：神经网络的拟合与梯度传播

代码位于 `code/q1_neural_network/`。实验要求、结果提交与分析题详见[实验文档](lab2.pdf)。

### 代码结构

- `models.py`：MLP 模型。
- `train.py`：训练实现。
- `main.py`：实验入口。
- `utils/`：设备选择、数据加载、梯度诊断、结果保存与绘图工具；设备选择位于 `device.py`，梯度诊断位于 `diagnostics.py`。
- `assets/`：固定数据和模型初值。
- `tests/`：公开接口自检。

请补全 `models.py` 和 `train.py` 中的 TODO，具体要求见代码注释。保留已有公共接口，其余代码无需修改。

### 运行

激活环境后，从本 README 所在目录进入第一题代码目录：

```bash
cd code/q1_neural_network
```

本节以下命令均在该代码目录执行：

```bash
python -m pytest -q
python main.py --task expression --device cpu
python main.py --task gradient --device cpu
```

公开测试抽查三种网络结构能否严格加载课程初值；完整实验会加载全部八组初值。测试通过不代表前向和训练实现正确，仍需运行两项完整实验。数据与初值由入口程序自动加载。

### 输出

结果保存在第一题代码目录下的 `outputs/expression/` 和 `outputs/gradient/`，包含指标、训练记录、模型参数和图像。重复运行可通过 `--output` 指定新的输出目录。

已有结果可直接重新绘图：

```bash
python -m utils.plot outputs/gradient
```

<a id="q2"></a>

## 第二题：无监督聚类

代码位于 `code/q2_clustering/`。实验要求、结果提交与分析题详见[实验文档](lab2.pdf)。

### 代码结构

- `submission.py`：需要补全的五个算法函数与三个评价函数。
- `main.py`：读取配置，运行拟合、评价与选择，保存结果。
- `train.py`、`evaluate.py`：算法迭代、随机重启、指标调度和跨种子选择。
- `extras.py`：自检、分阶段运行、小数据调试、评价回放和输出检查。
- `utils/`：数据读取、设备配置和结果保存工具。
- `config.json`、`data/`：实验配置与固定特征数据。

请补全 `submission.py` 中的八个 TODO，使用 PyTorch 实现并保留已有公共接口，具体要求见代码注释。模块职责、接口约定和调试方法见[代码框架说明](code/q2_clustering/FRAMEWORK_V0.md)。

### 运行

激活环境后，从本 README 所在目录进入第二题代码目录：

```bash
cd code/q2_clustering
```

本节以下命令均在该代码目录执行：

```bash
python extras.py self-check --device cpu
python main.py --device cpu
python extras.py check-output --output outputs_final
```

自检使用小型例子，完整实验通过 `main.py` 运行。数据自动加载，实验参数在 `config.json` 中设置。使用 GPU 时，将前两条命令中的 `--device cpu` 改为 `--device cuda`。

### 输出

结果保存在第二题代码目录下的 `outputs_final/`，包含候选指标、跨种子统计、选择记录、模型参数和最终簇编号 `clusters.npy`。簇编号与原始数据逐行对应。

运行后查看 `manifest.json` 确认完成状态，并用 `extras.py check-output` 检查输出格式与来源记录。该命令需要本地完整输出目录，不能单独检查提交的 `clusters.npy`。将核验结果写入报告，运行产物只需上传最终 `clusters.npy`；完整输出目录、实际运行代码和日志在本地保留至验收结束。该检查不评价聚类效果。重复运行时，通过 `--output` 指定新的空输出目录。

### 额外功能

额外功能统一放在 `extras.py`。以下命令仍在第二题代码目录执行，每次使用新的输出目录。

**分阶段运行。** 算法补完后可先用 `fit` 保存候选，评价函数补完后再用 `evaluate` 评价和选择，无需重新拟合：

```bash
python extras.py run --stage fit --device cpu \
  --config config.json --output outputs_fit
python extras.py evaluate --device cpu \
  --experiment outputs_fit --output outputs_evaluated
python extras.py check-output --output outputs_evaluated
```

**调整评价设置。** 将 `outputs_fit/config.json` 另存为 `config_eval.json`，修改主指标后对已有候选重新评价。例如将 `selection_metric` 改为 `calinski_harabasz`，再执行：

```bash
python extras.py evaluate --device cpu \
  --experiment outputs_fit --config config_eval.json --output outputs_evaluated_ch
python extras.py check-output --output outputs_evaluated_ch
```

主指标可选 `silhouette`、`calinski_harabasz` 或 `davies_bouldin`；回放时也可调整 Silhouette 抽样大小、抽样种子和线程数。修改算法或拟合参数后需要重新拟合，回放沿用原数据、设备和精度。

**小数据调试。** 准备只含特征矩阵 `X` 的 `toy_features.npz` 和匹配的 `config_dev.json` 后：

```bash
python extras.py run --device cpu --development \
  --input toy_features.npz --config config_dev.json --output outputs_dev
python extras.py check-output --development --input toy_features.npz --output outputs_dev
```

该模式用于小型合成数据调试，不能作为正式实验提交。数据与配置的生成示例见[代码框架说明](code/q2_clustering/FRAMEWORK_V0.md#小数据开发模式)。

其他可选功能：

- 分开自检：在 `extras.py self-check` 后加 `--part algorithms` 或 `--part metrics`，分别检查算法或评价函数。
- 切换精度：默认 `float32`，`extras.py self-check` 和 `extras.py run` 可加 `--dtype float64` 使用双精度。
- 替换实现：自检、扩展运行和评价回放均可加 `--implementation my_submission.py`，从指定文件读取算法与评价函数。正式运行前须将最终实现完整同步回 `submission.py`，并用该文件重新生成和核验最终输出。
- 候选标签归档：运行后自动生成 `candidate_clusters.npz`，可用 `np.load(..., allow_pickle=False)` 按候选名读取簇编号，方便比较不同运行。

完整参数可通过 `python main.py --help`、`python extras.py --help` 及各子命令的 `--help` 查看；精度、实现替换和归档读取的完整示例见[代码框架说明](code/q2_clustering/FRAMEWORK_V0.md#7-辅助功能)。

<a id="platform-107"></a>

## 107 平台配置

通过 Slurm 申请计算节点后运行 GPU 实验。在登录节点准备代码、创建 Conda 环境和提交作业；完整实验在计算节点执行。

将下面脚本保存为 `run_lab2_q2.sbatch`。替换工作目录和 Conda 初始化脚本的路径，确认 `ai26` 环境已按上文创建：

```bash
#!/bin/bash
#SBATCH --job-name=lab2_q2
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -e
cd "/path/to/lab2/code"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ai26
python -m pip install -r requirements-gpu.txt
python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
cd q2_clustering
python extras.py self-check --device cuda
python main.py --device cuda --output outputs_107
python extras.py check-output --output outputs_107
```

提交作业并查看状态：

```bash
sbatch run_lab2_q2.sbatch
squeue -u "$USER"
```

`--time=04:00:00` 是作业超时上限，不是实验预计耗时；可按实际需要申请更短时长。需要指定 GPU 型号时，可将 `--gres=gpu:1` 改为 `--gres=gpu:RTX5090:1` 或 `--gres=gpu:A100:1`，以账户可用资源为准。
作业结束后检查 Slurm 日志、输出目录中的 `manifest.json` 和 `extras.py check-output` 的文件检查结果。再次运行时换一个空输出目录。

第一题也可使用上述作业配置：环境检查完成后，将 `cd q2_clustering` 及其后的实验命令替换为：

```bash
cd q1_neural_network
python -m pytest -q
python main.py --task expression --device cuda
python main.py --task gradient --device cuda
```

## 代码框架说明

[第二题代码框架说明](code/q2_clustering/FRAMEWORK_V0.md) 介绍模块职责、函数接口和完整输出协议。
