import torch
import math
from typing import Optional, Callable


# [cite_start]=== 1. 从讲义中复制的 SGD 优化器实现 [cite: 858-877] ===
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 获取状态
                state = self.state[p]
                t = state.get("t", 0)

                grad = p.grad.data

                # 更新公式: theta = theta - (lr / sqrt(t+1)) * grad
                # 注意：讲义里的公式带有 lr decay (alpha / sqrt(t+1))
                p.data -= group["lr"] / math.sqrt(t + 1) * grad

                # 更新迭代次数
                state["t"] = t + 1

        return loss


# === 2. 实验主逻辑 ===


def run_experiment(learning_rate):
    print(f"\n=== Testing Learning Rate: {learning_rate} ===")

    # 固定随机种子，确保每次实验初始权重完全一样，只比较 LR 的影响
    torch.manual_seed(42)

    # [cite_start]初始化权重 (从讲义示例: 5 * randn) [cite: 884]
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

    # 初始化优化器
    opt = SGD([weights], lr=learning_rate)

    # [cite_start]运行 10 个 Step [cite: 898]
    losses = []
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()  # 简单的 loss 函数: w^2
        loss.backward()
        opt.step()
        losses.append(loss.item())

        print(f"Step {t + 1}: Loss = {loss.item():.6f}")

    return losses


# === 3. 执行对比 ===
if __name__ == "__main__":
    lrs_to_test = [1e-1, 1e-2, 1e-3]
    results = {}

    for lr in lrs_to_test:
        results[lr] = run_experiment(lr)

    # 简单打印对比结论
    print("\n=== Summary (Loss at Step 10) ===")
    for lr, loss_history in results.items():
        print(f"LR {lr}: Final Loss = {loss_history[-1]:.6f}")
