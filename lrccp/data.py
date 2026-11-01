import numpy as np, os
import torch
from torch.utils.data import Dataset

CACHE = os.path.expanduser('~/data/cubelearn/cache/hgr_ant0.npy')
META  = os.path.expanduser('~/data/cubelearn/cache/hgr_ant0_meta.npy')
SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')   # the official CubeLearn split CSVs, copied into the repo

class HGR(Dataset):
    def __init__(self, split):
        self.cache = np.load(CACHE, mmap_mode='r')                # (2880, 10, 128, 256) complex64
        self.meta  = np.load(META)                                # (2880, 3), columns are [user, gesture, sample]
        user, sample = self.meta[:, 0], self.meta[:, 2]

        csv = lambda name: np.loadtxt(os.path.join(SPLIT, f'{name}_samples.csv'), dtype=int)
        mask = {
            'train':  np.isin(user, range(6)) & np.isin(sample, csv('train')),
            'val':    np.isin(user, range(6)) & np.isin(sample, csv('validation')),
            'test':   np.isin(user, range(6)) & np.isin(sample, csv('test')),
            'outset': np.isin(user, [6, 7]),                      # unseen users, all of their samples
        }[split]
        self.idx = np.where(mask)[0]

    def __len__(self):
        return len(self.idx)                                      # computed from the mask, not hard-coded

    def __getitem__(self, k):
        i = self.idx[k]
        x = torch.from_numpy(np.array(self.cache[i]))             # actually reads from the mmap and gives a writable copy
        return torch.view_as_real(x), int(self.meta[i, 1])        # (10, 128, 256, 2) float32, last dim is [real, imag]
