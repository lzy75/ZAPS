"""
固定正交 db4 二维多级离散小波变换 W（torch 原生、可微）

用途（论文 Eq.22）：将 log 先验的 Hessian 对角化近似为
    ∂²log p_t / ∂x_t²  ≈  W D_t W^⊤
其中 W 为正交 DWT，D_t 为可学习对角阵。ZAPS 中需要计算
    W D_t W^⊤ v  =  synthesis( D_t ⊙ analysis(v) )

关键点：
  - 必须可微（D_t、v 都要反传），故不能用 numpy 版 pywt；这里仅用 pywt
    取出 db4 滤波器系数，变换本身用 torch FFT 循环卷积实现。
  - analysis 记为正交算子 A，synthesis 严格取 A 的伴随（FFT 域取共轭），
    由 db4 正交性保证完美重构 synthesis(analysis(x)) = x。
  - 系数按 Mallat 布局打包成与输入同形状 [.,H,W]，使 D_t 可写成 [C,H,W]
    的逐系数对角（全对角，对应论文 'uniformly across diagonals'）。
"""
import torch
import torch.nn as nn
import pywt


class OrthogonalDWT2D(nn.Module):
    """正交 db4 多级 2D DWT，周期延拓，完美重构

    参数:
        wave  : 小波名，论文用 'db4'
        level : 分解级数（输入边长需整除 2^level）
    """

    def __init__(self, wave: str = "db4", level: int = 3):
        super().__init__()
        self.level = level
        # 仅取滤波器系数（一次性），变换全程用 torch 实现
        dec_lo, dec_hi, _, _ = pywt.Wavelet(wave).filter_bank
        self.register_buffer("h", torch.tensor(dec_lo, dtype=torch.float32))  # 低通
        self.register_buffer("g", torch.tensor(dec_hi, dtype=torch.float32))  # 高通

    # ── FFT 循环卷积及其伴随（沿最后一维）──────────────────
    def _cconv(self, x, filt):
        N = x.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(x, n=N) * torch.fft.rfft(filt, n=N), n=N)

    def _cconv_adj(self, x, filt):
        N = x.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(x, n=N) * torch.fft.rfft(filt, n=N).conj(), n=N)

    def _up(self, a, N):
        shp = list(a.shape); shp[-1] = N
        z = torch.zeros(shp, dtype=a.dtype, device=a.device)
        z[..., ::2] = a
        return z

    # ── 1D 单级（沿最后一维）──────────────────────────────
    def _dwt1d(self, x):
        return self._cconv(x, self.h)[..., ::2], self._cconv(x, self.g)[..., ::2]

    def _idwt1d(self, lo, hi):
        N = lo.shape[-1] * 2
        return self._cconv_adj(self._up(lo, N), self.h) + self._cconv_adj(self._up(hi, N), self.g)

    # ── 2D 单级（可分离：先 W 向再 H 向）──────────────────
    def _dwt2d(self, x):
        lo, hi = self._dwt1d(x)                              # 沿 W
        a, b = self._dwt1d(lo.transpose(-1, -2))            # 沿 H
        c, d = self._dwt1d(hi.transpose(-1, -2))
        return (a.transpose(-1, -2), b.transpose(-1, -2),
                c.transpose(-1, -2), d.transpose(-1, -2))   # LL, LH, HL, HH

    def _idwt2d(self, a, b, c, d):
        lo = self._idwt1d(a.transpose(-1, -2), b.transpose(-1, -2)).transpose(-1, -2)
        hi = self._idwt1d(c.transpose(-1, -2), d.transpose(-1, -2)).transpose(-1, -2)
        return self._idwt1d(lo, hi)

    # ── 多级正/逆变换（Mallat 打包到同形状）────────────────
    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,C,H,W] → 打包系数 [B,C,H,W]（= A x）"""
        packed = torch.empty_like(x)
        cur = x
        h_, w_ = x.shape[-2], x.shape[-1]
        for _ in range(self.level):
            ll, lh, hl, hh = self._dwt2d(cur)
            h2, w2 = h_ // 2, w_ // 2
            packed[..., :h2, w2:w_]   = lh
            packed[..., h2:h_, :w2]   = hl
            packed[..., h2:h_, w2:w_] = hh
            packed[..., :h2, :w2]     = ll   # 末级 LL 留在左上，期间会被下一级覆盖
            cur, h_, w_ = ll, h2, w2
        return packed

    def synthesis(self, packed: torch.Tensor) -> torch.Tensor:
        """打包系数 [B,C,H,W] → x [B,C,H,W]（= A^⊤ packed）"""
        H, W = packed.shape[-2], packed.shape[-1]
        sizes = []
        h_, w_ = H, W
        for _ in range(self.level):
            sizes.append((h_, w_)); h_ //= 2; w_ //= 2
        hs, ws = sizes[-1][0] // 2, sizes[-1][1] // 2
        cur = packed[..., :hs, :ws]
        for h_, w_ in reversed(sizes):
            h2, w2 = h_ // 2, w_ // 2
            cur = self._idwt2d(
                cur,
                packed[..., :h2, w2:w_],
                packed[..., h2:h_, :w2],
                packed[..., h2:h_, w2:w_],
            )
        return cur

    # ── 完美重构自检 ──────────────────────────────────────
    @torch.no_grad()
    def self_check(self, size: int = 64, channels: int = 3) -> float:
        x = torch.randn(1, channels, size, size, device=self.h.device)
        err = (self.synthesis(self.analysis(x)) - x).abs().max().item()
        return err
