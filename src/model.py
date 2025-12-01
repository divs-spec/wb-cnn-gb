# src/model.py

import torch
import torch.nn as nn
from torchvision import models

class ResNet18Regressor(nn.Module):
    """ResNet-18 backbone (pretrained) with a custom regression head for 2 outputs."""
    def __init__(self, pretrained=True):
        super().__init__()
        # Load ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)
        # Remove original final layer (fc)
        in_features = self.backbone.fc.in_features  # typically 512
        self.backbone.fc = nn.Identity()  # remove classifier
        
        # Regression head (can be more complex if desired)
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # outputs: Temperature, Tint
        )
    
    def forward(self, x):
        x = self.backbone(x)  # output shape [batch, 512]
        out = self.regressor(x)  # [batch, 2]
        return out
