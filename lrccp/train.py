import argparse, os, csv, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from .data import HGR
from .models import DT2DCNN


def run_epoch(model, loader, device, optimizer=None):
    """跑一遍 loader。给 optimizer 就是训练，不给就是评估。返回 (平均 loss, accuracy)。"""
    train = optimizer is not None
    model.train(train)                       # True → train 模式，False → eval 模式
    crit = nn.CrossEntropyLoss()
    tot_loss, tot_correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):      # 评估时不算梯度，省显存
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
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--out', default='runs/dt_fixed_s0')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    loaders = {s: DataLoader(HGR(s), batch_size=32, shuffle=(s == 'train'), num_workers=0)
               for s in ['train', 'val', 'test', 'outset']}

    model = DT2DCNN(trainable=args.trainable).to(device)
    opt = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=3e-4)

    best_val, log = -1, []
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, loaders['train'], device, opt)
        va_loss, va_acc = run_epoch(model, loaders['val'], device)
        log.append([ep, tr_loss, tr_acc, va_loss, va_acc])
        print(f'ep {ep:2d}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}  {time.perf_counter()-t0:.0f}s', flush=True)
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), os.path.join(args.out, 'best.pt'))

    with open(os.path.join(args.out, 'log.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']); w.writerows(log)

    # 最后：读回最佳模型，test 和 outset 各跑一次，只跑一次
    model.load_state_dict(torch.load(os.path.join(args.out, 'best.pt')))
    for s in ['test', 'outset']:
        loss, acc = run_epoch(model, loaders[s], device)
        print(f'{s}: acc {acc:.4f}', flush=True)

if __name__ == '__main__':
    main()