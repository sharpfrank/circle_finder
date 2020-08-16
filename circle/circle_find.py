import torch
from .circlenet import CircleNet
import numpy as np


def find_circle(circle_parameters_model: CircleNet, image: np.array, device: torch.device) -> (float, float, float):
    """ Find a circle in a noisy image
    parameters:
    circle_parameter_model - a pytorch model
    image - a image represented as a pytorch array
    device - gpu or cpu
    results: the location and size of the circle within the image (row, col, radius)
    """
    # Prepare the input image for use by the model predictor
    b_image = torch.from_numpy(image).to(device)
    b_image = b_image.unsqueeze(dim=0).unsqueeze(dim=0)
    # Predict the location and size of the circle in the image.
    circle_parameters = circle_parameters_model(b_image)[0]
    return circle_parameters
