import sys, os
sys.path.append(os.path.expanduser('~/cubelearn'))   # 让 python 找得到作者的 network.py
import numpy as np
import torch
from network import Range_Fourier_Net
from cplxmodule.nn.modules.casting import TensorToCplx

a = np.load(os.path.expanduser('~/data/cubelearn/hand_organized/0_0_0.npy'))
x_np = (a[0, 0, :, 0, :] + 1j * a[1, 0, :, 0, :]).astype(np.complex64)   # (128, 256)

# 参照：numpy 沿 sample 轴做 FFT
ref = np.fft.fft(x_np, axis=1)

# 被测：作者的网络层
net = Range_Fourier_Net()
x_t = TensorToCplx()(torch.view_as_real(torch.from_numpy(x_np)))   # numpy 复数 → cplxmodule 的 Cplx 类型
with torch.no_grad():
    y = net(x_t)
y_np = (y.real + 1j * y.imag).numpy()           # Cplx → numpy 复数

print('shape:', y_np.shape)
print('max abs diff:', np.abs(y_np - ref).max())
print('ref max:', np.abs(ref).max())            # 给 diff 一个参照尺度