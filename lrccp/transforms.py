import torch, torch.nn as nn

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