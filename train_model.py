import torch
import torch.optim as optimizer
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from circle.circlenet import circlenet18, circlenet9
from circle.circle_dataset import CircleParmDataset
from circle.circle_train import train_model
import argparse

parser = argparse.ArgumentParser(description='Train a circle finding model')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--train_size', type=int, default=10000)
parser.add_argument('--val_size', type=int, default=2000)
parser.add_argument('--model_name', type=str, default='circle_model.pk')
parser.add_argument('--model_type', type=str, default='circlenet18')

args = parser.parse_args()


def main(cm_args):

    batch_size = cm_args.batch_size
    num_epochs = cm_args.epochs
    model_name = cm_args.model_name
    train_size = cm_args.train_size
    val_size = cm_args.val_size
    model_type = cm_args.model_type

    # problem constants, should not change
    image_size = 200
    circle_max_radius = 50
    noise_level = 2

    # create the training and validation sets
    train_set = CircleParmDataset(train_size, image_size, circle_max_radius, noise_level)
    val_set = CircleParmDataset(val_size, image_size, circle_max_radius, noise_level)
    print(f'train set size: {len(train_set)} validation set size: {len(val_set)}')
    print(f'batch size: {batch_size}')

    data_loaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    }

    # create an empty model
    if model_type == 'circlenet18':
        model = circlenet18(num_classes=3)
    elif model_type == 'circlenet9':
        model = circlenet9(num_classes=3)
    else:
        print(f'Invalid model selected: {model_type}')
        return

    print(f'model type: {model_type}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # setup optimizer and learning rate scheduler
    optimizer_ft = optimizer.Adam(model.parameters(), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

    print(f'Begin training model {model_name}')
    model = train_model(model, data_loaders, optimizer_ft, exp_lr_scheduler,
                        device=device, num_epochs=num_epochs)

    print(f'Training complete. Saving model {model_name}')
    torch.save(model.state_dict(), model_name)


main(args)
