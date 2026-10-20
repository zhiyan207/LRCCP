import numpy as np, os
SRC = os.path.expanduser('~/data/cubelearn/hand_organized')
cache = np.load(os.path.expanduser('~/data/cubelearn/cache/hgr_ant0.npy'), mmap_mode='r')
meta  = np.load(os.path.expanduser('~/data/cubelearn/cache/hgr_ant0_meta.npy'))

rng = np.random.default_rng(0)
for i in rng.choice(len(cache), 5, replace=False):
    u, g, s = i // 360, (i % 360) // 30, i % 30
    assert (meta[i] == [u, g, s]).all(), f'meta mismatch at {i}'
    a = np.load(os.path.join(SRC, f'{u}_{g}_{s}.npy'), mmap_mode='r')
    ref = a[0, :, :, 0, :] + 1j * a[1, :, :, 0, :]
    ok = np.array_equal(np.array(cache[i]), ref.astype(np.complex64))
    print(i, (u, g, s), 'OK' if ok else 'MISMATCH')