import sys, os, torch
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.models import DT2DCNN

def drift(layer, ref_layer):
    """One number for DFT; for Butterfly, a list with one number per level plus the total."""
    if hasattr(layer, 'levels'):                     # Butterfly
        per = [ (w - w0).norm() / w0.norm() for w, w0 in zip(layer.W, ref_layer.W) ]
        num = sum(((w - w0).norm() ** 2 for w, w0 in zip(layer.W, ref_layer.W)))
        den = sum((w0.norm() ** 2 for w0 in ref_layer.W))
        return [p.item() for p in per], (num / den).sqrt().item()
    return (layer.W - ref_layer.W).norm() / ref_layer.W.norm()

if len(sys.argv) < 2:
    sys.exit('usage: python experiments/weight_drift.py <run prefix>, e.g. dt_learn_lpp3e-3')
prefix = sys.argv[1]                               # prefix of this batch of runs, e.g. dt_learn_lpp3e-3
found = 0
for s in range(3):
    ckpt = f'runs/{prefix}_s{s}/best.pt'
    if not os.path.exists(ckpt):                  # skip seeds that haven't finished yet
        continue
    found += 1
    sd = torch.load(ckpt, map_location='cpu')
    transform = 'butterfly' if 'range_dft.W.0' in sd else 'dft'   # Butterfly is saved as W.0 … W.7, DFT has a single W
    ref = DT2DCNN(transform=transform)            # untrained, so W is the FFT init
    m = DT2DCNN(trainable=True, transform=transform)
    m.load_state_dict(sd)
    for name in ['range_dft', 'doppler_dft']:
        d = drift(getattr(m, name), getattr(ref, name))
        if isinstance(d, tuple):
            per, total = d
            print(f'seed {s}  {name:12s}  drift = {total:.4f}  per level ' + ' '.join(f'{p:.3f}' for p in per))
        else:
            print(f'seed {s}  {name:12s}  drift = {d:.4f}')
if found == 0:
    print(f'no runs/{prefix}_s*/best.pt found, check the prefix')
