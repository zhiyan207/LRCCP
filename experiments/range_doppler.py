import numpy as np
import matplotlib
matplotlib.use('Agg')        # WSL 没显示器，画完存成图片
import matplotlib.pyplot as plt
import os

a = np.load(os.path.expanduser('~/data/cubelearn/hand_organized/0_0_0.npy'))   # (2, 10, 128, 12, 256)

def rd_map(a, frame, ant):
    I = a[0, frame, :, ant, :]
    Q = a[1, frame, :, ant, :]

    x = I + 1j * Q

    R = np.fft.fft(x, axis=1)
    RD = np.fft.fft(R, axis=0)
    RD = np.fft.fftshift(RD, axes=0)

    mag = 20 * np.log10(np.abs(RD) + 1e-6)
    return mag

plt.figure(figsize=(20, 6))
for f in range(10):
    plt.subplot(2, 5, f+1)
    plt.imshow(rd_map(a, f, 0)[:, :40], aspect='auto', origin='lower', vmin=40, vmax=140)
    plt.title(f'frame {f}')
plt.savefig(os.path.expanduser('~/LRCCP/experiments/range_doppler.png'))