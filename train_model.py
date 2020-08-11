import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary
from circle.circlenet import circlenet18, circlenet9
from circle.circle_dataset import CircleParmDataset
from circle.circle_train import train_model

image_size = 200
circle_max_radius = 50
noise_level = 2

batch_size = 500
num_epochs = 100

train_set = CircleParmDataset(10000, image_size, circle_max_radius, noise_level)
val_set = CircleParmDataset(2000, image_size, circle_max_radius, noise_level)

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = circlenet18(num_classes=3)
model.to(device)

summary(model, input_size=(1, 200, 200))


optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, 
    num_epochs=num_epochs, device=device)


model_name = 'circlenet18-exp.pk'
torch.save(model.state_dict(), model_name)