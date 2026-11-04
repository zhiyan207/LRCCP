import sys, os, torch
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.models import DT2DCNN
from lrccp.metrics import drift

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
        layer, ref_layer = getattr(m, name), getattr(ref, name)
        per, total = drift(list(layer.parameters()), list(ref_layer.parameters()))   # Butterfly: W[0] … W[7]; DFT: [W]
        if hasattr(layer, 'levels'):                 # Butterfly
            print(f'seed {s}  {name:12s}  drift = {total:.4f}  per level ' + ' '.join(f'{p:.3f}' for p in per))
        else:
            print(f'seed {s}  {name:12s}  drift = {total:.4f}')
if found == 0:
    print(f'no runs/{prefix}_s*/best.pt found, check the prefix')
