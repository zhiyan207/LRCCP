import torch

def butterfly_level(x, w, b):
    """一级蝶形。x: (N,) 复数；w: (N/b, 2, 2, b/2)；b: 这一级的块大小。返回 (N,)。"""
    N = x.shape[0]
    xp = x.reshape(N//b, 2, b//2)                                 # 空 1：x 切成 N/b 个块，每块上下两半，shape (N/b, 2, b/2)
    top, bot = xp[:, 0], xp[:, 1]                # 各 (N/b, b/2)
    y_top = w[:, 0, 0] * top + w[:, 0, 1] * bot  # (N/b, b/2)
    y_bot = w[:, 1, 0] * top + w[:, 1, 1] * bot                               # 空 2
    return torch.stack([y_top, y_bot], dim=1).reshape(N)   # 拼回 (N/b, 2, b/2) 再拉平

def fft_weights(N, b):
    """b 点蝶形那一级的 FFT 系数，复制到 N/b 个块。返回 (N/b, 2, 2, b/2)。"""
    i = torch.arange(b // 2)
    tw = torch.exp(-1j * 2 * torch.pi * i / b)                                  # 空 1：ω_b^i，b 点的 twiddle
    ones = torch.ones(b // 2, dtype=torch.complex64)
    w = torch.stack([torch.stack([ones, tw]), torch.stack([ones, -tw])])   # (2, 2, b/2)
    return w.unsqueeze(0).repeat(N // b, 1, 1, 1)                                # 空 2：在前面加块维度并复制 N/b 份，shape (N/b, 2, 2, b/2)

def bit_reversal(N):
    bits = N.bit_length() - 1
    return torch.tensor([int(f'{k:0{bits}b}'[::-1], 2) for k in range(N)])

def fft_butterfly(x):
    N = x.shape[0]
    x = x[bit_reversal(N)]                       # 先置换
    b = 2
    while b <= N:
        x = butterfly_level(x, fft_weights(N, b), b)
        b *= 2
    return x

x = torch.randn(256, dtype=torch.complex64)
print((fft_butterfly(x) - torch.fft.fft(x)).abs().max() / torch.fft.fft(x).abs().max())


N = 8
x = torch.arange(N, dtype=torch.float32).to(torch.complex64)
i = torch.arange(N // 2)
tw = torch.exp(-2j * torch.pi * i / N)
ones = torch.ones(N // 2, dtype=torch.complex64)
w8 = torch.stack([torch.stack([ones, tw]), torch.stack([ones, -tw])])[None]   # (1, 2, 2, 4)，[None] 加块维度

E = torch.fft.fft(x[0::2]); O = torch.fft.fft(x[1::2])
y = butterfly_level(torch.cat([E, O]), w8, b=8)
print((y - torch.fft.fft(x)).abs().max())