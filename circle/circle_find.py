import torch


def find_circle(circle_parameters_model, image, device):
    """find a circle in a noisy image"""
    b_image = torch.from_numpy(image).to(device)
    b_image = b_image.unsqueeze(dim=0).unsqueeze(dim=0)
    y = circle_parameters_model(b_image)[0]
    return y
