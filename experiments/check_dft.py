import sys, os, numpy as np, torch
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.transforms import DFT

cache = np.load(os.path.expanduser('~/data/cubelearn/cache/hgr_ant0.npy'), mmap_mode='r')
x = torch.from_numpy(np.array(cache[0]))          # (10, 128, 256) complex64

def rel_err(a, b):
    return (a - b).abs().max().item() / b.abs().max().item()

with torch.no_grad():
    # range 轴：最后一维，直接喂
    y   = DFT(256)(x)
    ref = torch.fft.fft(x, dim=-1)
    print('range   rel err:', rel_err(y, ref))

    # Doppler 轴：chirp 轴在倒数第二维，转到最后、做完再转回来
    y   = DFT(128, centered=True)(x.transpose(-1, -2)).transpose(-1, -2)
    ref = torch.fft.fftshift(torch.fft.fft(x, dim=-2), dim=-2)
    print('doppler rel err:', rel_err(y, ref))

    # range → Doppler 串起来，和 models.py 里 forward 的写法一模一样，输出是 (10, 256 range, 128 doppler)
    y   = DFT(128, centered=True)(DFT(256)(x).transpose(-1, -2))
    ref = torch.fft.fftshift(torch.fft.fft2(x), dim=-2).transpose(-1, -2)
    print('RD map  rel err:', rel_err(y, ref))
