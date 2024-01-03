import torch
import torch.nn as nn




class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16').eval()
        #model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=1000)
        # self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=embedding_dim)
        # dim_mlp = self.backbone.fc.in_features
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

        self.encoder = model
        self.fc = nn.Linear(1000, embedding_dim)

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
