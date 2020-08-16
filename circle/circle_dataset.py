from torch.utils.data import Dataset
from .circle_generator import noisy_circle
import numpy as np


class CircleParmDataset(Dataset):
    """" Image generation source for a circle in a noise image"""
    def __init__(self, count: int, size: int, radius: int, noise_level: float):
        """
        parameters:
        count - the number of images to generate.
        size - the size of the images to generate in pixels, same number of rows as columns.
        radius - the radius of the circle to generate, value in pixels.
        noise_level - the noise level to add to the image, uniformly distributed from 1 to noise_level.
        """
        self.images = []
        self.circle_parameters = []
        for _ in range(count):
            circle_parameters, circle_image = noisy_circle(size, radius, noise_level)
            self.images.append(circle_image)
            self.circle_parameters.append(circle_parameters)

    def __len__(self) -> int:
        """ Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx) -> (np.array, np.array):
        """ Return an image and its parameters using idx as an index 0..count."""
        r_image = np.expand_dims(self.images[idx], axis=0)
        parameters = np.array(self.circle_parameters[idx], dtype=np.float32)
        return [r_image, parameters]
