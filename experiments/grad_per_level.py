
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lrccp.data import HGR
from lrccp.models import DT2DCNN


def mean_cos(vecs):
    cs = []
    for a in range(len(vecs)):
        for b in range(a + 1, len(vecs)):
            u, v = vecs[a], vecs[b]
            cs.append((torch.vdot(u, v).real / (u.norm() * v.norm())).item())
    return sum(cs) / len(cs)


def main():
    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DT2DCNN(
        transform='butterfly',
        trainable=True
    ).to(device)

    model.train()
    train_loader = DataLoader(
        HGR('train'),
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    crit = nn.CrossEntropyLoss()

    range_grads = [[] for _ in model.range_dft.W]
    doppler_grads = [[] for _ in model.doppler_dft.W]
    range_vecs = [[] for _ in model.range_dft.W]
    doppler_vecs = [[] for _ in model.doppler_dft.W]

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= 5:
            break

        x, y = x.to(device), y.to(device)

        model.zero_grad(set_to_none=True)

        out = model(x)
        loss = crit(out, y)
        loss.backward()

        # Range Butterfly
        for i, p in enumerate(model.range_dft.W):
            relative_grad = (
                p.grad.norm() / p.detach().norm()
            )
            range_grads[i].append(relative_grad.item())
            range_vecs[i].append(p.grad.flatten().clone())

        # Doppler Butterfly
        for i, p in enumerate(model.doppler_dft.W):
            relative_grad = (
                p.grad.norm() / p.detach().norm()
            )
            doppler_grads[i].append(relative_grad.item())
            doppler_vecs[i].append(p.grad.flatten().clone())

    if not range_grads[0]:
        raise RuntimeError("No training batches were loaded")

    print("\nRange Butterfly:")
    for i, values in enumerate(range_grads):
        avg = sum(values) / len(values)
        b = model.range_dft.levels[i]
        print(f"  b={b}  relative_grad={avg:.6f}  cos={mean_cos(range_vecs[i]):.3f}")

    print("\nDoppler Butterfly:")
    for i, values in enumerate(doppler_grads):
        avg = sum(values) / len(values)
        b = model.doppler_dft.levels[i]
        print(f"  b={b}  relative_grad={avg:.6f}  cos={mean_cos(doppler_vecs[i]):.3f}")


if __name__ == '__main__':
    main()
