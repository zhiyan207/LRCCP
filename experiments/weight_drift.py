import sys, os, torch
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.models import DT2DCNN

def drift(layer, ref_layer):
    """对 DFT 返回一个数；对 Butterfly 返回每级一个数的列表加一个总的。"""
    if hasattr(layer, 'levels'):                     # Butterfly
        per = [ (w - w0).norm() / w0.norm() for w, w0 in zip(layer.W, ref_layer.W) ]
        num = sum(((w - w0).norm() ** 2 for w, w0 in zip(layer.W, ref_layer.W)))
        den = sum((w0.norm() ** 2 for w0 in ref_layer.W))
        return [p.item() for p in per], (num / den).sqrt().item()
    return (layer.W - ref_layer.W).norm() / ref_layer.W.norm()

if len(sys.argv) < 2:
    sys.exit('用法: python experiments/weight_drift.py <run 前缀>，比如 dt_learn_lpp3e-3')
prefix = sys.argv[1]                               # 这批 run 的前缀，比如 dt_learn_lpp3e-3
found = 0
for s in range(3):
    ckpt = f'runs/{prefix}_s{s}/best.pt'
    if not os.path.exists(ckpt):                  # 这批还没跑完的 seed 先跳过
        continue
    found += 1
    sd = torch.load(ckpt, map_location='cpu')
    transform = 'butterfly' if 'range_dft.W.0' in sd else 'dft'   # 蝶形存成 W.0 … W.7，DFT 只有一个 W
    ref = DT2DCNN(transform=transform)            # 没训过的，W 就是 FFT 初值
    m = DT2DCNN(trainable=True, transform=transform)
    m.load_state_dict(sd)
    for name in ['range_dft', 'doppler_dft']:
        d = drift(getattr(m, name), getattr(ref, name))
        if isinstance(d, tuple):
            per, total = d
            print(f'seed {s}  {name:12s}  drift = {total:.4f}  每级 ' + ' '.join(f'{p:.3f}' for p in per))
        else:
            print(f'seed {s}  {name:12s}  drift = {d:.4f}')
if found == 0:
    print(f'没找到 runs/{prefix}_s*/best.pt，检查一下前缀')
