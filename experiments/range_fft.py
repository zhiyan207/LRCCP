import numpy as np
import matplotlib
matplotlib.use('Agg')        # WSL 没显示器，画完存成图片
import matplotlib.pyplot as plt
import os

a = np.load(os.path.expanduser('~/data/cubelearn/hand_organized/0_0_0.npy'))   # (2, 10, 128, 12, 256)

# 空 1：取 frame 0, chirp 0, virtual antenna 0 的 I 路和 Q 路，各是长度 256 的向量
I = a[0, 0, 0, 0, :]
Q = a[1, 0, 0, 0, :]

# 空 2：合成一个复数向量
x = I + Q * 1j

X = np.fft.fft(x)
mag = np.abs(X)

# 空 3：peak 在第几个 bin？用 np.argmax
print('peak bin:', np.argmax(mag))

plt.plot(mag)
plt.xlabel('range bin'); plt.ylabel('|X|')
plt.savefig('/home/yuzhiyan/LRCCP/experiments/range_fft.png')