import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class RidgeletLineDataset(Dataset):
    def __init__(self, p: int, samples_per_class: int = 100):
        self.p = p
        self.samples_per_class = samples_per_class
        self.total_samples = (p + 1) * samples_per_class
    def __len__(self):
        return self.total_samples
    def __getitem__(self, idx):
        k = idx // self.samples_per_class
        l = np.random.randint(0, self.p)
        img = np.zeros((self.p, self.p), dtype=np.float32)
        if k < self.p:
            for i in range(self.p):
                j = (k * i + l) % self.p
                img[i, j] = 1.0
        else:
            img[:, l] = 1.0
        tensor_img = torch.from_numpy(img).unsqueeze(0)
        return tensor_img, torch.tensor(k, dtype=torch.long)
