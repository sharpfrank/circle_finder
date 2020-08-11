import torch

def find_circle(circle_parms_model, image, device):
    b_image = torch.from_numpy(image).to(device)
    b_image = b_image.unsqueeze(dim=0).unsqueeze(dim=0)
    y = circle_parms_model(b_image)[0]
    return y