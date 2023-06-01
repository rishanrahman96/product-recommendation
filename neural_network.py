from torchvision import models
import torch
import torch.nn as nn


class Neural_networkN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        for name, module in self.resnet50.named_modules():
            if name == 'layer4' or name == 'fc':
                for param in module.parameters():
                    param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False


        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(1000, 13)
        self.main = nn.Sequential(self.resnet50, self.linear)
        
    def forward(self, inp):
        x = self.main(inp)
        return x