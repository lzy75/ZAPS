"""数值检查 SuperResolutionOperator.transpose 是否真的是 H 的伴随。"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import torch

from modules.degradations import SuperResolutionOperator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    operator = SuperResolutionOperator(scale_factor=4, noise_sigma=0.0).to(device)
    x = torch.randn(1, 3, 256, 256, device=device)
    y = torch.randn_like(operator.H(x))

    lhs = torch.sum(operator.H(x) * y)
    rhs = torch.sum(x * operator.transpose(y))
    relative_error = (lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12)

    # ZAPS 需要梯度穿过 H^T；除了伴随恒等式，也检查这条路径没有被截断。
    y_grad = y.detach().requires_grad_(True)
    operator.transpose(y_grad).square().mean().backward()
    gradient_ok = y_grad.grad is not None and torch.isfinite(y_grad.grad).all().item()

    print(f"device: {device}")
    print(f"<Hx,y>: {lhs.item():.9f}")
    print(f"<x,H^Ty>: {rhs.item():.9f}")
    print(f"内积伴随相对误差: {relative_error.item():.9e}")
    print(f"H^T 梯度链可用: {gradient_ok}")
    if relative_error.item() >= 1e-5 or not gradient_ok:
        raise SystemExit("FAIL：超分算子的精确伴随检查未通过")
    print("PASS：transpose 是 H 的精确伴随，且支持 ZAPS 展开反向传播")


if __name__ == "__main__":
    main()
