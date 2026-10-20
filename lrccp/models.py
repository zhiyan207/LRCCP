import torch, torch.nn as nn, torch.nn.functional as F
from .transforms import DFT

class DT2DCNN(nn.Module):
    """CubeLearn 的 DT_2DCNN，第一层换成自己的 DFT。
    trainable=False → Baseline 1（固定 FFT）；True → Baseline 2（可学习稠密）。"""
    def __init__(self, trainable=False):
        super().__init__()
        self.range_dft   = DFT(256, trainable=trainable, centered=False)                       # 空 1：256 点，沿 sample 轴
        self.doppler_dft = DFT(128, trainable=trainable, centered=True)                       # 空 2：128 点，零频居中
        self.conv1 = nn.Conv2d(1, 4, 3);  self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3);  self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3); self.bn3 = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d((1, 2), ceil_mode=True)
        self.fc1 = nn.Linear(960, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 12)

    def forward(self, x):                               # (B, 10, 128, 256, 2) float32
        x = torch.view_as_complex(x.contiguous())                                      # 空 3：变回 complex64
        x = self.range_dft(x)                           # (B, 10, 128, 256)
        x = x.transpose(-1, -2)                                      # 空 4：chirp 轴转到最后
        x = self.doppler_dft(x)                         # (B, 10, 256, 128)
        x = torch.abs(x)                                      # 空 5：取模，变实数
        x = x.sum(dim=2)                                      # 空 6：sum 掉 range 轴 → (B, 10, 128)
        x = x.reshape(-1, 1, 10, 128)                   # 加一个 channel 维给 Conv2d
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
            x = self.pool(F.relu(bn(conv(x))))
        x = x.reshape(x.shape[0], -1)                   # (B, 960)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)