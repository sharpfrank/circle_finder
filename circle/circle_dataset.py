from torch.utils.data import Dataset
from .circle_generator import noisy_circle
import numpy as np


class CircleParmDataset(Dataset):
    def __init__(self, count, size, radius, noise):
        self.images = []
        self.circle_parameters = []
        for _ in range(count):
            circle_parameters, circle_image = noisy_circle(size, radius, noise)
            self.images.append(circle_image)
            self.circle_parameters.append(circle_parameters)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        r_image = np.expand_dims(self.images[idx], axis=0)
        parameters = np.array(self.circle_parameters[idx], dtype=np.float32)
        return [r_image, parameters]
