import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16').eval()
model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=1000)
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        # self.encoder = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more convolutional layers as needed
       # )
        self.encoder = model
        self.fc = nn.Linear(1000, embedding_dim)
        #self.fc = nn.Linear(8192, embedding_dim)

    def forward_one(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_one(anchor)
        output_positive = self.forward_one(positive)
        output_negative = self.forward_one(negative)
        return output_anchor, output_positive, output_negative
