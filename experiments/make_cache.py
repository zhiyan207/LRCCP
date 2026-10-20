import numpy as np, os, time

SRC = os.path.expanduser('~/data/cubelearn/hand_organized')
DST = os.path.expanduser('~/data/cubelearn/cache/hgr_ant0.npy')
META = os.path.expanduser('~/data/cubelearn/cache/hgr_ant0_meta.npy')
os.makedirs(os.path.dirname(DST), exist_ok=True)

N_U, N_G, N_S = 8, 12, 30
N = N_U * N_G * N_S
ANT = 0

data = np.lib.format.open_memmap(DST, mode='w+', dtype=np.complex64, shape=(N, 10, 128, 256))
meta = np.zeros((N, 3), dtype=np.int16)

t0 = time.perf_counter()
for u in range(N_U):
    for g in range(N_G):
        for s in range(N_S):
            i = u * 360 + g * 30 + s
            a = np.load(os.path.join(SRC, f'{u}_{g}_{s}.npy'), mmap_mode='r')
            data[i] = a[0, :, :, ANT, :] + 1j * a[1, :, :, ANT, :]                             # 天线 ANT 的 I + jQ，shape (10,128,256)
            meta[i] = [u, g, s]
            if i % 100 == 0:
                print(f'{i}/{N}  {time.perf_counter()-t0:.0f}s', flush=True)

data.flush()
np.save(META, meta)
print('done', data.shape, data.dtype)