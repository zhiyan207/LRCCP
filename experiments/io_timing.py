import numpy as np, os, time
p = os.path.expanduser('~/data/cubelearn/hand_organized/0_0_0.npy')

t = time.perf_counter()
a = np.load(p)
x = a[0, :, 0, 0, :] + 1j * a[1, :, 0, 0, :]
print('full load + slice: %.3f s' % (time.perf_counter() - t))

t = time.perf_counter()
a = np.load(p, mmap_mode='r')
x = a[0, :, 0, 0, :] + 1j * a[1, :, 0, 0, :]                       # 同样的切法
x = np.array(x)                  # mmap 切片还是"懒"的，强制真读出来
print('mmap slice:        %.3f s' % (time.perf_counter() - t))