"""
诊断16: 从 ckpt 权重 shape 反推 ImageNet 模型真实训练配置,与我们 MODEL_CONFIG 比对。

背景(最强证据):
  · 4张完全不同的ImageNet图, recon全锁15.8~16.6(方差仅0.3), recon_TV 40-56>>GT_TV。
    "不管输入什么图输出都一样烂" = 固定数值/配置错, 非先验能力(先验弱会随图波动)。
  · 已排除采样全家桶/伴随/fp16/值域/单步去噪(38dB)。ζ=0引导暴跌(引导在帮忙)。FFHQ全对。
  · 头号嫌疑: ImageNetDiffusionModel.MODEL_CONFIG 与官方256x256_diffusion_uncond.pt
    训练FLAGS不一致 → 模型结构错位 → 系统性偏(但小t容错高故单步还有38dB)。
  · load_state_dict strict默认True没报错=键匹配(非丢权重), 但config语义错位仍可能崩输出。

本脚本: 不建模型, 直接 torch.load ckpt, 从权重 shape **反推真实配置**(比查README硬):
  - num_channels: input_blocks.0.0.weight 的 out_channels
  - learn_sigma: out.2.weight 的 out_channels 是否 = 2×3
  - channel_mult/深度: 扫 input_blocks.*.0.*weight 的通道倍增序列
  - attention层: 含 'attention'/'.qkv.'/'proj_out' 的键出现在哪些 input_blocks 层 → 反推 attention_resolutions
  - num_res_blocks: 每个分辨率级的 resblock 数
  然后打印我们 MODEL_CONFIG, 逐项对照, 标不一致。
  对 FFHQ 也做一遍(作对照, 它是对的)。

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_ckpt_config.py
"""
import os, sys, re
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from configs.config import MODEL_DIR
from modules.diffusion_model import FFHQDiffusionModel, ImageNetDiffusionModel

CKPTS = {
    "imagenet": (os.path.join(MODEL_DIR, "256x256_diffusion_uncond.pt"), ImageNetDiffusionModel.MODEL_CONFIG),
    "ffhq":     (os.path.join(MODEL_DIR, "ffhq_10m.pt"), FFHQDiffusionModel.MODEL_CONFIG),
}


def analyze(name, path, cfg):
    print(f"\n{'='*70}\n  {name}  从权重反推真实配置\n{'='*70}", flush=True)
    if not os.path.exists(path):
        print(f"  [跳过] ckpt不存在 {path}", flush=True)
        return
    sd = torch.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        print("  [异常] ckpt不是state_dict", flush=True); return
    keys = list(sd.keys())
    print(f"  权重键总数={len(keys)}", flush=True)

    # num_channels: 第一个 input_blocks 卷积 out_channels
    for k in keys:
        if re.search(r"input_blocks\.0\.0\.weight", k):
            print(f"  [num_channels] {k}: shape={tuple(sd[k].shape)} → 反推 num_channels={sd[k].shape[0]}  (config={cfg['num_channels']})", flush=True)
            break
    # learn_sigma: 输出层 out_channels 是否 2×3=6
    for k in keys:
        if re.search(r"\bout\.2\.weight$", k) or k.endswith("out.2.weight"):
            oc = sd[k].shape[0]
            print(f"  [learn_sigma] {k}: out_channels={oc} → {'2C=6=learn_sigma✓' if oc==6 else f'={oc}'}  (config={cfg['learn_sigma']})", flush=True)
            break
    # 通道倍增序列(channel_mult) & 深度: 收集所有 input_blocks.N.0 卷积的 out_channels
    mults = {}
    for k in keys:
        m = re.search(r"input_blocks\.(\d+)\.0\.(?:weight|op\.weight)$", k)
        if m and sd[k].dim() == 4:
            mults[int(m.group(1))] = sd[k].shape[0]
    if mults:
        seq = [mults[i] for i in sorted(mults)]
        base = seq[0] if seq else 1
        print(f"  [channel序列] input_blocks各层out_ch={seq}", flush=True)
        print(f"                除以base({base})≈channel_mult={[round(s/base,1) for s in seq]}  (config channel_mult='{cfg['channel_mult']}')", flush=True)
    # attention层位置: 含 qkv 的 input_blocks 层号
    attn_blocks = sorted(set(int(m.group(1)) for k in keys
                             for m in [re.search(r"input_blocks\.(\d+)\..*(qkv|attention)", k)] if m))
    print(f"  [attention出现在 input_blocks 层] {attn_blocks}  (config attention_resolutions='{cfg['attention_resolutions']}')", flush=True)
    # 是否 class conditional: 有无 label_emb
    has_label = any("label_emb" in k for k in keys)
    print(f"  [class_cond] label_emb存在={has_label}  (config class_cond={cfg['class_cond']}) {'✗不一致!' if has_label!=cfg['class_cond'] else '✓'}", flush=True)


def main():
    for name, (path, cfg) in CKPTS.items():
        analyze(name, path, cfg)
    print("\n判读:", flush=True)
    print("  · 反推的 num_channels/channel序列/attention层/learn_sigma 与 config 不一致 ⇒ 找到根因!", flush=True)
    print("    结构错位会让整个ImageNet系统性偏(小t容错高故单步仍38dB),FFHQ config对故正常。", flush=True)
    print("  · 全部一致 ⇒ 排除结构, ImageNet与FFHQ反推结果对比看有无隐藏差异。", flush=True)


if __name__ == "__main__":
    main()
