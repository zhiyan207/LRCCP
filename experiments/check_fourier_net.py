import sys, os
sys.path.append(os.path.expanduser('~/cubelearn'))   # so Python can find the authors' network.py
import numpy as np
import torch
from network import Range_Fourier_Net
from cplxmodule.nn.modules.casting import TensorToCplx

a = np.load(os.path.expanduser('~/data/cubelearn/hand_organized/0_0_0.npy'))
x_np = (a[0, 0, :, 0, :] + 1j * a[1, 0, :, 0, :]).astype(np.complex64)   # (128, 256)

# reference: numpy FFT along the sample axis
ref = np.fft.fft(x_np, axis=1)

# under test: the authors' network layer
net = Range_Fourier_Net()
x_t = TensorToCplx()(torch.view_as_real(torch.from_numpy(x_np)))   # numpy complex → cplxmodule Cplx
with torch.no_grad():
    y = net(x_t)
y_np = (y.real + 1j * y.imag).numpy()           # Cplx → numpy complex

print('shape:', y_np.shape)
print('max abs diff:', np.abs(y_np - ref).max())
print('ref max:', np.abs(ref).max())            # gives the diff a scale