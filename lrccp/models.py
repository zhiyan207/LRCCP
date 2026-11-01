import torch, torch.nn as nn, torch.nn.functional as F
from .transforms import DFT, Butterfly

class DT2DCNN(nn.Module):
    """CubeLearn's DT_2DCNN with the first layer replaced by our own transform.
    transform='dft':       trainable=False → fixed FFT; True → learnable dense complex layer (CubeLearn)
    transform='butterfly': trainable=False → FFT (butterfly-factorized, identical output); True → learnable butterfly (ours)"""
    def __init__(self, trainable=False, transform='dft'):
        super().__init__()
        T = {'dft': DFT, 'butterfly': Butterfly}[transform]
        self.range_dft   = T(256, trainable=trainable)                       # 256 points, along the sample axis
        self.doppler_dft = T(128, trainable=trainable, centered=True)                       # 128 points, zero frequency centered
        self.conv1 = nn.Conv2d(1, 4, 3);  self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3);  self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3); self.bn3 = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d((1, 2), ceil_mode=True)
        self.fc1 = nn.Linear(960, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 12)

    def forward(self, x):                               # (B, 10, 128, 256, 2) float32
        x = torch.view_as_complex(x.contiguous())                                      # back to complex64
        x = self.range_dft(x)                           # (B, 10, 128, 256)
        x = x.transpose(-1, -2)                                      # move the chirp axis last
        x = self.doppler_dft(x)                         # (B, 10, 256, 128)
        x = torch.abs(x)                                      # take the magnitude, now real
        x = x.sum(dim=2)                                      # sum out the range axis → (B, 10, 128)
        x = x.reshape(-1, 1, 10, 128)                   # add a channel dim for Conv2d
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
            x = self.pool(F.relu(bn(conv(x))))
        x = x.reshape(x.shape[0], -1)                   # (B, 960)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)