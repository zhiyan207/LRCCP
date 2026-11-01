import torch, torch.nn as nn

def butterfly_level(x, w, b):
    """One butterfly level. x: (..., N); w: (N/b, 2, 2, b/2). Returns (..., N)."""
    N = x.shape[-1]
    xp = x.reshape(*x.shape[:-1], N // b, 2, b // 2)
    top, bot = xp[..., 0, :], xp[..., 1, :]
    y_top = w[:, 0, 0] * top + w[:, 0, 1] * bot
    y_bot = w[:, 1, 0] * top + w[:, 1, 1] * bot
    return torch.stack([y_top, y_bot], dim=-2).reshape(*x.shape[:-1], N)

def fft_weights(N, b):
    """FFT coefficients of the b-point butterfly level, copied into N/b blocks. Returns (N/b, 2, 2, b/2)."""
    i = torch.arange(b // 2)
    tw = torch.exp(-1j * 2 * torch.pi * i / b)
    ones = torch.ones(b // 2, dtype=torch.complex64)
    w = torch.stack([torch.stack([ones, tw]), torch.stack([ones, -tw])])   # (2, 2, b/2)
    return w.unsqueeze(0).repeat(N // b, 1, 1, 1)

def bit_reversal(N):
    bits = N.bit_length() - 1
    return torch.tensor([int(f'{k:0{bits}b}'[::-1], 2) for k in range(N)])

class Butterfly(nn.Module):
    """Radix-2 butterfly transform along the last dim: log2(n) levels, 2n complex parameters per level.
    Initialized to the FFT coefficients. trainable=False is exactly the FFT; True is a learnable butterfly. Same interface as DFT."""
    def __init__(self, n, trainable=False, centered=False):
        super().__init__()
        self.n, self.centered = n, centered
        self.levels = [2 ** k for k in range(1, n.bit_length())]                      # one b per level, from 2 to n, kept as a plain list for forward
        self.W = nn.ParameterList([
            nn.Parameter(fft_weights(n, b), requires_grad=trainable)                                 # one Parameter per level, initialized with fft_weights(n, b), requires_grad=trainable
            for b in self.levels])
        self.register_buffer('perm', bit_reversal(n))   # the permutation isn't a parameter, but it has to move with the model on .to(device)

    def forward(self, x):                          # x: (..., n) complex64
        x = x[..., self.perm]                                 # bit-reversal
        for b, w in zip(self.levels, self.W):
            x = butterfly_level(x, w, b)
        if self.centered:
            x = torch.fft.fftshift(x, dim=-1)
        return x

class DFT(nn.Module):
    """n-point DFT along the last dim. trainable=False is the fixed FFT,
    True is a learnable dense complex layer initialized to the DFT (as in CubeLearn)."""
    def __init__(self, n, trainable=False, centered=False):
        super().__init__()
        k = torch.arange(n)
        if centered:
            k = (k + n // 2) % n            
        W = torch.exp(-2j * torch.pi * k[:, None] * torch.arange(n)[None, :] / n).to(torch.complex64)
        self.W = nn.Parameter(W, requires_grad=trainable)

    def forward(self, x):                 # x: (..., n) complex64
        return x @ self.W.T         