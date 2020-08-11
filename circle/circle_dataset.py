from torch.utils.data import Dataset, DataLoader
from circle_generator import draw_circle, noisy_circle
import numpy as np

class CircleParmDataset(Dataset):
    def __init__(self, count, size, radius, noise):
        self.images = []
        self.circle_parms = []
        for _ in range(count):
            circle_parms, circle_image = noisy_circle(size, radius, noise)
            self.images.append(circle_image)
            self.circle_parms.append(circle_parms)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        r_image = np.expand_dims(self.images[idx], axis=0)
        parms = np.array(self.circle_parms[idx], dtype=np.float32)
        return [r_image, parms]
