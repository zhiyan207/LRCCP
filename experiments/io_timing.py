import numpy as np, os, time
p = os.path.expanduser('~/data/cubelearn/hand_organized/0_0_0.npy')

t = time.perf_counter()
a = np.load(p)
x = a[0, :, 0, 0, :] + 1j * a[1, :, 0, 0, :]
print('full load + slice: %.3f s' % (time.perf_counter() - t))

t = time.perf_counter()
a = np.load(p, mmap_mode='r')
x = a[0, :, 0, 0, :] + 1j * a[1, :, 0, 0, :]                       # same slicing
x = np.array(x)                  # mmap slices are lazy; force an actual read
print('mmap slice:        %.3f s' % (time.perf_counter() - t))