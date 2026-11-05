"""检查可用设备，并为实验选择默认设备。"""

import torch


def check_device_available(device) -> bool:
    """检查设备是否可用，CPU 默认可用。"""
    if device == "cuda" and not torch.cuda.is_available():
        return False
    if device == "mps" and not torch.backends.mps.is_available():
        return False
    if device == "xpu" and not torch.xpu.is_available():
        return False
    return True


def auto_device() -> str:
    """按 CUDA、MPS、XPU、CPU 的顺序选择可用设备。"""
    for device in ["cuda", "mps", "xpu", "cpu"]:
        if check_device_available(device):
            return device
    return "cpu"
