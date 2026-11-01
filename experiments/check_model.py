import sys, os, torch
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.models import DT2DCNN

x = torch.randn(2, 10, 128, 256, 2)        # fake batch of 2, same shape the dataset returns
for transform in ['dft', 'butterfly']:
    m = DT2DCNN(transform=transform)
    y = m(x)
    print(transform, y.shape)              # expect (2, 12)
    print(sum(p.numel() for p in m.parameters() if p.requires_grad), 'trainable params')