import numpy as np
import matplotlib
matplotlib.use('Agg')        # no display on WSL, so save the plot to a file
import matplotlib.pyplot as plt
import os

a = np.load(os.path.expanduser('~/data/cubelearn/hand_organized/0_0_0.npy'))   # (2, 10, 128, 12, 256)

# I and Q of frame 0, chirp 0, virtual antenna 0, each a length-256 vector
I = a[0, 0, 0, 0, :]
Q = a[1, 0, 0, 0, :]

# combine them into one complex vector
x = I + Q * 1j

X = np.fft.fft(x)
mag = np.abs(X)

# which bin is the peak in? use np.argmax
print('peak bin:', np.argmax(mag))

plt.plot(mag)
plt.xlabel('range bin'); plt.ylabel('|X|')
plt.savefig('/home/yuzhiyan/LRCCP/experiments/range_fft.png')