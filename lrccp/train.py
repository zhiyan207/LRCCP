import argparse, os, csv, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from .data import HGR
from .models import DT2DCNN
from .metrics import drift


def run_epoch(model, loader, device, optimizer=None):
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates. Returns (mean loss, accuracy)."""
    train = optimizer is not None
    model.train(train)                       # True → train mode, False → eval mode
    crit = nn.CrossEntropyLoss()
    tot_loss, tot_correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):      # no gradients during eval, saves GPU memory
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * len(y)
            tot_correct += out.argmax(dim=1).eq(y).sum().item()
            n += len(y)
    return tot_loss / n, tot_correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trainable', action='store_true')
    p.add_argument('--transform', default='dft')
    p.add_argument('--init', default='fft')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lpp_lr', type=float, default=1e-3)
    p.add_argument('--out', default='runs/dt_fixed_s0')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    loaders = {s: DataLoader(HGR(s), batch_size=32, shuffle=(s == 'train'), num_workers=0)
               for s in ['train', 'val', 'test', 'outset']}

    model = DT2DCNN(trainable=args.trainable, transform=args.transform, init=args.init).to(device)
    layers = {'range': model.range_dft, 'doppler': model.doppler_dft}
    ref = {k: [p.detach().clone() for p in L.parameters()] for k, L in layers.items()}
    dft_params  = list(model.range_dft.parameters()) + list(model.doppler_dft.parameters())
    dft_ids     = {id(p) for p in dft_params}
    base_params = [p for p in model.parameters() if id(p) not in dft_ids]
    opt = torch.optim.Adam([{'params': base_params},
                        {'params': dft_params, 'lr': args.lpp_lr}], lr=3e-4)

    best_val, log = -1, []
    levels_log = []
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, loaders['train'], device, opt)
        va_loss, va_acc = run_epoch(model, loaders['val'], device)
        range_per, range_total = drift(
            layers['range'].parameters(),
            ref['range']
        )
        doppler_per, doppler_total = drift(
            layers['doppler'].parameters(),
            ref['doppler']
        )
        log.append([ep, tr_loss, tr_acc, va_loss, va_acc, range_total, doppler_total])
        levels_log.append([ep] + range_per + doppler_per)
        print(f'ep {ep:2d}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}  {time.perf_counter()-t0:.0f}s', flush=True)
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), os.path.join(args.out, 'best.pt'))

    with open(os.path.join(args.out, 'log.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'range_drift', 'doppler_drift']); w.writerows(log)

    if hasattr(model.range_dft, 'levels'):
        range_cols = [f'range_b{b}' for b in model.range_dft.levels]
    else:
        range_cols = ['range_W']
    if hasattr(model.doppler_dft, 'levels'):
        doppler_cols = [f'doppler_b{b}' for b in model.doppler_dft.levels]
    else:
        doppler_cols = ['doppler_W']

    with open(os.path.join(args.out, 'drift_levels.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch'] + range_cols + doppler_cols)
        w.writerows(levels_log)

    # finally: reload the best model and run test and outset once each, and only once
    model.load_state_dict(torch.load(os.path.join(args.out, 'best.pt')))
    for s in ['test', 'outset']:
        loss, acc = run_epoch(model, loaders[s], device)
        print(f'{s}: acc {acc:.4f}', flush=True)

if __name__ == '__main__':
    main()