import sys, os, torch
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.models import DT2DCNN

m = DT2DCNN()
x = torch.randn(2, 10, 128, 256, 2)        # 假装一个 batch=2 的输入，和 dataset 交出来的 shape 一样
y = m(x)
print(y.shape)                             # 期望 (2, 12)
print(sum(p.numel() for p in m.parameters() if p.requires_grad), 'trainable params')