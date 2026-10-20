import sys, os
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.data import HGR
from torch.utils.data import DataLoader

for s in ['train', 'val', 'test', 'outset']:
    print(s, len(HGR(s)))

x, y = next(iter(DataLoader(HGR('train'), batch_size=32, shuffle=True)))
print(x.shape, x.dtype, y.shape, y.dtype, y[:8])