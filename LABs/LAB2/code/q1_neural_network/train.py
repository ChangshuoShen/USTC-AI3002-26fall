import torch

from utils.diagnostics import gradient_profile

UPDATER = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

def train(model, x, y, *, steps, learning_rate, optimizer="sgd", diagnostics=False):
    # 根据实验配置创建优化器，使用模型参数和固定学习率。
    updater = UPDATER[optimizer](
        model.parameters(), lr=learning_rate)
    history = []
    for step in range(steps + 1):
        # 记录完成 step 次更新后的状态；step=0 对应初始模型。
        if step % 100 == 0 or step == steps:
            # 这里只记录损失，不保留本次前向的计算图。
            with torch.no_grad():
                row = {"step": step, "train_mse": float((model(x) - y).square().mean())}
            if diagnostics and step in {0, steps}:
                # 首尾用固定的 128 个样本统计梯度，供 RMS 可视化使用。
                row["gradient_profile"] = gradient_profile(model, x[:128], y[:128])
            history.append(row)
        # 最后一次只记录结果，确保总共执行 steps 次更新。
        if step == steps:
            break

        # 此处需要补全
        # TODO：在此实现一次更新，使用当前模型、优化器和全部 x、y。
        raise NotImplementedError("请补全单次训练更新，并删除这一行占位异常。")
        # 补全位置结束
    return history
