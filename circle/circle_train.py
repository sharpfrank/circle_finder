import torch
import time
import copy
from collections import defaultdict
from .circlenet import CircleNet
from .circle_dataset import CircleParmDataset
from torch.optim import lr_scheduler
import numpy as np


def calc_loss(predict: np.array, target: np.array, metrics: {}) -> torch.tensor:
    model_loss = torch.nn.MSELoss()
    mse = model_loss(predict, target)
    metrics['loss'] += mse.data.cpu().numpy() * target.size(0)
    return mse


def print_metrics(metrics: {}, epoch_samples: int, phase: str):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    


def train_model(model: CircleNet, data_loaders: {CircleParmDataset, CircleParmDataset},
                optimizer: torch.optim, scheduler: lr_scheduler,
                device: torch.device, num_epochs: int = 25) -> CircleNet:
    """ Train a model to find a circle in a noisy image
    parameters:
    data_loader - a source for images, contains both training and validation images.
    optimizer - a pytorch based optimizer for use in training the model
    scheduler - a learning rate scheduler.
    device = model will use this device, cpu or gpu.
    num_epochs - the number of epochs to train for.
    returns: Return the best model as determined by validation loss.
      Also, print a log of training and validation losses.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('-' * 64)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('compute time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
