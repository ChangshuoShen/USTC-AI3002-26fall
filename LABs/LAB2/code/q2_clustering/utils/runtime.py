"""统一设置 PyTorch 设备、浮点精度和数值检查容差。"""
import os

import torch

DTYPES = ("float32", "float64")
DTYPE_TOLERANCES = {
    "float32": dict(rtol=1e-5, atol=1e-6),
    "float64": dict(rtol=1e-7, atol=1e-9),
}


DEVICE_ENV = "LAB2_DEVICE"
VALID_DEVICES = ("cpu", "cuda")
device = torch.device("cpu")
dtype = torch.float32


def configure_dtype(selected="float32"):
    global dtype
    if selected not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    dtype = getattr(torch, selected)
    return dtype


def check_tolerances():
    return dict(DTYPE_TOLERANCES[str(dtype).removeprefix("torch.")])


def normalization_tolerances():
    tolerances = check_tolerances()
    # 大对数密度下的归一化会损失少量精度，单精度检查需留出舍入容差。
    if dtype == torch.float32:
        tolerances["rtol"] = 1e-4
    return tolerances


def resolve_device(explicit=None, inherited=None):
    value = explicit or os.environ.get(DEVICE_ENV) or inherited or "cpu"
    value = str(value).strip().lower()
    if value not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {value!r}")
    return value


def validate_device(selected):
    selected = resolve_device(selected)
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is false")
    return dict(device=selected, torch=torch.__version__,
                cuda=torch.version.cuda if selected == "cuda" else None,
                device_name=torch.cuda.get_device_name(0) if selected == "cuda" else None)


def configure_device(explicit=None, inherited=None):
    global device
    selected = resolve_device(explicit, inherited)
    validate_device(selected)
    device = torch.device("cuda:0" if selected == "cuda" else "cpu")
    return device


def configure(selected, precision, threads):
    """先设置设备与精度，再加载算法与评价实现，保证其 device 属性正确。"""
    configure_device(selected)
    configure_dtype(precision)
    torch.set_num_threads(threads)
    os.environ[DEVICE_ENV] = device.type
    return validate_device(device.type)
