"""观察激活前梯度，不改写训练参数的 .grad。"""

import torch
from torch import nn


def gradient_profile(model, x, y):
    """计算隐藏层激活前梯度的 RMS、相对输出端的比例及局部导数均值。"""
    preactivations = []
    handles = [layer.register_forward_hook(lambda module, inputs, output: preactivations.append(output))
               for layer in model.hidden]
    try:
        prediction = model(x)
        loss = (prediction - y).square().mean()
        gradients = torch.autograd.grad(loss, [*preactivations, prediction])
    finally:
        for handle in handles:
            handle.remove()
    output_rms = float(gradients[-1].detach().square().mean().sqrt())
    layers = []
    for index, (z, gradient) in enumerate(zip(preactivations, gradients[:-1])):
        # 激活函数的局部导数，仅用于解释，不代替实测梯度。
        if isinstance(model.activation, nn.Sigmoid):
            a = z.detach().sigmoid()
            derivative = a * (1 - a)
        else:
            derivative = (z.detach() > 0).float()
        value = float(gradient.detach().square().mean().sqrt())
        layers.append({"layer": index + 1, "gradient_rms": value,
                       "relative_to_output": value / output_rms if output_rms else None,
                       "derivative_mean": float(derivative.mean())})
    return {"output_gradient_rms": output_rms, "layers": layers}
