import torch
N = 8
x = torch.arange(N, dtype=torch.float32).to(torch.complex64)      # 随便一个输入
xp = x.reshape(2, N // 2)
top, bot = xp[0], xp[1]

i = torch.arange(N // 2)
tw = torch.exp(-2j * torch.pi * i / N)           # ω^i，四个 twiddle
w = torch.stack([torch.stack([torch.ones(4), tw]),
                 torch.stack([torch.ones(4), -tw])])   # (2, 2, 4)

y_top = w[0, 0] * top + w[0, 1] * bot
y_bot = w[1, 0] * top + w[1, 1] * bot
print(y_top, y_bot)

E = torch.fft.fft(x[0::2])          # 偶数位的 4 点 FFT
O = torch.fft.fft(x[1::2])          # 奇数位的 4 点 FFT
y_top = w[0, 0] * E + w[0, 1] * O
y_bot = w[1, 0] * E + w[1, 1] * O
y = torch.cat([y_top, y_bot])
ref = torch.fft.fft(x)
print((y - ref).abs().max())