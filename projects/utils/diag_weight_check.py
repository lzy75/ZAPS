"""
诊断20: ImageNet 权重完整性检查——验证"底层模型/权重坏了"的怀疑。

背景(强怀疑模型层):
  · 1000步标准DDPM无条件生成 TV减小但图仍"一坨没规律的东西"(用户肉眼)。
    正常guided-diffusion ImageNet模型即使宽分布,1000步无条件也该出有物体轮廓/结构的自然图。
  · 之前"排除模型"的两条证据都不够硬:
    - 单步去噪38dB: 可能只是对轻微加噪做平庸恒等映射, 不证明学到ImageNet分布。
    - ckpt反推架构正确: 只证结构对/能加载, 不证权重【数值本身】没损坏(下载截断/坏权重)。

本脚本(纯查权重, 不采样): 对 imagenet + ffhq 两个ckpt:
  A. 文件大小(官方256x256_uncond约2.1GB, ffhq_10m约300+MB) + MD5(供与官方比对)
  B. 权重数值健康度: 遍历所有张量, 统计 有无NaN/Inf、全零层数、每层std分布、
     整体权重std范围。坏权重特征: 大量NaN/Inf、大片全零、std异常(过大过小或=0)。
  C. 两模型对照: FFHQ是好的(29.78), 对比ImageNet权重健康度有无异常。
  D. 关键层抽样: 打印 input_blocks.0 / middle_block / out 几个关键层的 mean/std,
     和FFHQ对应层量级对比(同为guided-diffusion,量级应相近)。

判据:
  · ImageNet有NaN/Inf/大片全零/std异常 而FFHQ正常 ⇒ 权重损坏, 重新下载256x256_diffusion_uncond.pt
  · 两者权重健康度都正常 ⇒ 权重没坏, 问题在更微妙处(需进一步查)

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_weight_check.py
"""
import os, sys, hashlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from configs.config import MODEL_DIR

CKPTS = {
    "imagenet": os.path.join(MODEL_DIR, "256x256_diffusion_uncond.pt"),
    "ffhq":     os.path.join(MODEL_DIR, "ffhq_10m.pt"),
}
# 官方参考(用于比对): 256x256_diffusion_uncond.pt 官方大小约 2.1GB
KNOWN = {
    "imagenet": "openai官方256x256_diffusion_uncond.pt ≈ 2.1GB (2211383297 bytes附近)",
    "ffhq":     "ffhq_10m.pt (DPS/ILVR用) ≈ 357MB附近",
}


def md5_head(path, nbytes=None):
    """整文件MD5(可能慢, 大文件只算前256MB做快速指纹)。"""
    h = hashlib.md5()
    read = 0
    cap = nbytes if nbytes else float("inf")
    with open(path, "rb") as f:
        while read < cap:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest(), read


def analyze(name, path):
    print(f"\n{'='*70}\n  {name}  权重完整性\n{'='*70}", flush=True)
    if not os.path.exists(path):
        print(f"  [跳过] 不存在 {path}", flush=True)
        return
    size = os.path.getsize(path)
    print(f"  文件: {path}", flush=True)
    print(f"  大小: {size:,} bytes ({size/1e9:.3f} GB)  参考: {KNOWN.get(name,'?')}", flush=True)
    # 前256MB快速指纹(整文件MD5太慢, 截断下载的话前部也可能变? 用整文件更可靠但慢——这里算整个)
    print(f"  计算整文件MD5(大文件稍慢)...", flush=True)
    full_md5, read = md5_head(path)
    print(f"  MD5(整文件)= {full_md5}   已读={read:,}", flush=True)

    sd = torch.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        print(f"  [异常] 不是state_dict: {type(sd)}", flush=True); return

    n_tensors = 0
    n_nan = n_inf = n_allzero = 0
    stds, means = [], []
    total_params = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        n_tensors += 1
        vf = v.detach().float()
        total_params += vf.numel()
        if torch.isnan(vf).any(): n_nan += 1
        if torch.isinf(vf).any(): n_inf += 1
        if float(vf.abs().max()) == 0.0: n_allzero += 1
        stds.append(float(vf.std()))
        means.append(float(vf.mean()))

    import statistics as st
    print(f"  张量数={n_tensors}  总参数={total_params:,}", flush=True)
    print(f"  含NaN的层={n_nan}  含Inf的层={n_inf}  全零层={n_allzero}", flush=True)
    if stds:
        print(f"  各层std: 均值={st.mean(stds):.4f} 中位={st.median(stds):.4f} "
              f"范围=[{min(stds):.4f}, {max(stds):.4f}]", flush=True)
        print(f"  各层mean绝对值最大={max(abs(m) for m in means):.4f}", flush=True)

    # 关键层抽样
    print(f"  关键层抽样:", flush=True)
    for pat in ["input_blocks.0.0.weight", "middle_block.0.in_layers.2.weight", "out.2.weight"]:
        for k, v in sd.items():
            if k.endswith(pat) and torch.is_tensor(v):
                vf = v.detach().float()
                print(f"    {k}: shape={tuple(vf.shape)} mean={vf.mean():.5f} std={vf.std():.5f} "
                      f"min={vf.min():.3f} max={vf.max():.3f}", flush=True)
                break


def main():
    for name, path in CKPTS.items():
        analyze(name, path)
    print(f"\n{'='*70}\n  判读\n{'='*70}", flush=True)
    print("  · ImageNet 大小明显小于~2.1GB ⇒ 下载截断, 重下。", flush=True)
    print("  · ImageNet 有NaN/Inf/大量全零层, 或各层std与FFHQ量级差很多 ⇒ 权重损坏。", flush=True)
    print("  · 两者健康度相近、大小正常、无NaN ⇒ 权重没坏, 排除本假设。", flush=True)
    print("  · 把ImageNet的MD5记下, 可与openai官方发布的MD5(或重下一份)比对确认。", flush=True)


if __name__ == "__main__":
    main()
