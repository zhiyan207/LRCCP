import sys, os, torch
sys.path.append(os.path.expanduser('~/LRCCP'))
from lrccp.models import DT2DCNN

if len(sys.argv) < 2:
    sys.exit('用法: python experiments/weight_drift.py <run 前缀>，比如 dt_learn_lpp3e-3')
prefix = sys.argv[1]                               # 这批 run 的前缀，比如 dt_learn_lpp3e-3
ref = DT2DCNN()                                   # 没训过的，W 就是 DFT 矩阵
found = 0
for s in range(3):
    ckpt = f'runs/{prefix}_s{s}/best.pt'
    if not os.path.exists(ckpt):                  # 这批还没跑完的 seed 先跳过
        continue
    found += 1
    m = DT2DCNN(trainable=True)
    m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    for name in ['range_dft', 'doppler_dft']:
        W0 = getattr(ref, name).W
        W  = getattr(m, name).W
        drift = (torch.linalg.norm(W - W0) / torch.linalg.norm(W0)).item()                            # 空：Frobenius 范数的比值 ‖W − W0‖ / ‖W0‖
        print(f'seed {s}  {name:12s}  drift = {drift:.4f}')
if found == 0:
    print(f'没找到 runs/{prefix}_s*/best.pt，检查一下前缀')
