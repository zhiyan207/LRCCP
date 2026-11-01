import sys, os, numpy as np, torch
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.transforms import Butterfly

cache = np.load(os.path.expanduser('~/data/cubelearn/cache/hgr_ant0.npy'), mmap_mode='r')
x = torch.from_numpy(np.array(cache[0]))          # (10, 128, 256) complex64

def rel_err(a, b):
    return (a - b).abs().max().item() / b.abs().max().item()

with torch.no_grad():
    # range axis: the last dim, fed in directly
    y   = Butterfly(256)(x)
    ref = torch.fft.fft(x, dim=-1)
    print('range   rel err:', rel_err(y, ref))

    # Doppler axis: chirps are the second-to-last dim; move them last, transform, move them back
    y   = Butterfly(128, centered=True)(x.transpose(-1, -2)).transpose(-1, -2)
    ref = torch.fft.fftshift(torch.fft.fft(x, dim=-2), dim=-2)
    print('doppler rel err:', rel_err(y, ref))

    # range then Doppler, written exactly as in forward in models.py; output is (10, 256 range, 128 doppler)
    y   = Butterfly(128, centered=True)(Butterfly(256)(x).transpose(-1, -2))
    ref = torch.fft.fftshift(torch.fft.fft2(x), dim=-2).transpose(-1, -2)
    print('RD map  rel err:', rel_err(y, ref))

print('params:', sum(p.numel() for p in Butterfly(256).parameters()))
