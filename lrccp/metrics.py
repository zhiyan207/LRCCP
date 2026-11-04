import torch

@torch.no_grad()
def drift(params, ref):
    """params, ref: lists of tensors (e.g. list(layer.parameters())), not a single tensor."""
    params, ref = list(params), list(ref)

    per = [
        (w - w0).norm() / w0.norm()
        for w, w0 in zip(params, ref)
    ]

    per = [p.item() for p in per]

    num = sum(
        (w - w0).norm() ** 2
        for w, w0 in zip(params, ref)
    )

    den = sum(w0.norm() ** 2 for w0 in ref)

    total = (num / den).sqrt().item()

    return (per, total)