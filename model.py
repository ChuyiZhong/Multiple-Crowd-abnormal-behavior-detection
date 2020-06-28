import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MCLNet(nn.Module):
    def __init__(self, num_cls):
        super(MCLNet, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_cls)

        self.net = model_ft
    
    def forward(self, x):
        x = self.net(x)
        return x
