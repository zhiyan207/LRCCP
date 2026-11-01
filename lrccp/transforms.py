import torch, torch.nn as nn

def butterfly_level(x, w, b):
    """一级蝶形。x: (..., N)；w: (N/b, 2, 2, b/2)。返回 (..., N)。"""
    N = x.shape[-1]
    xp = x.reshape(*x.shape[:-1], N // b, 2, b // 2)
    top, bot = xp[..., 0, :], xp[..., 1, :]
    y_top = w[:, 0, 0] * top + w[:, 0, 1] * bot
    y_bot = w[:, 1, 0] * top + w[:, 1, 1] * bot
    return torch.stack([y_top, y_bot], dim=-2).reshape(*x.shape[:-1], N)

def fft_weights(N, b):
    """b 点蝶形那一级的 FFT 系数，复制到 N/b 个块。返回 (N/b, 2, 2, b/2)。"""
    i = torch.arange(b // 2)
    tw = torch.exp(-1j * 2 * torch.pi * i / b)
    ones = torch.ones(b // 2, dtype=torch.complex64)
    w = torch.stack([torch.stack([ones, tw]), torch.stack([ones, -tw])])   # (2, 2, b/2)
    return w.unsqueeze(0).repeat(N // b, 1, 1, 1)

def bit_reversal(N):
    bits = N.bit_length() - 1
    return torch.tensor([int(f'{k:0{bits}b}'[::-1], 2) for k in range(N)])

class Butterfly(nn.Module):
    """沿最后一维的 radix-2 蝶形变换，log2(n) 级，每级 2n 个复数参数。
    FFT 系数初始化。trainable=False 就等于 FFT；True 是可学习蝶形。接口同 DFT。"""
    def __init__(self, n, trainable=False, centered=False):
        super().__init__()
        self.n, self.centered = n, centered
        self.levels = [2 ** k for k in range(1, n.bit_length())]                      # 空 1：b 从 2 到 n 每级一个 b，存成普通 list，forward 要用
        self.W = nn.ParameterList([
            nn.Parameter(fft_weights(n, b), requires_grad=trainable)                                 # 空 2：每级一个 Parameter，用 fft_weights(n, b) 初始化，requires_grad=trainable
            for b in self.levels])
        self.register_buffer('perm', bit_reversal(n))   # 置换不是参数，但要跟着模型一起 .to(device)

    def forward(self, x):                          # x: (..., n) complex64
        x = x[..., self.perm]                                 # 空 3：bit-reversal
        for b, w in zip(self.levels, self.W):
            x = butterfly_level(x, w, b)                             # 空 4
        if self.centered:
            x = torch.fft.fftshift(x, dim=-1)                             # 空 5
        return x

class DFT(nn.Module):
    """沿最后一维做 n 点 DFT。trainable=False 就是固定 FFT (Baseline 1), 
    True 就是从 DFT 出发的可学习稠密复数层 (Baseline 2)"""
    def __init__(self, n, trainable=False, centered=False):
        super().__init__()
        k = torch.arange(n)
        if centered:
            k = (k + n // 2) % n            
        W = torch.exp(-2j * torch.pi * k[:, None] * torch.arange(n)[None, :] / n).to(torch.complex64)
        self.W = nn.Parameter(W, requires_grad=trainable)

    def forward(self, x):                 # x: (..., n) complex64
        return x @ self.W.T         