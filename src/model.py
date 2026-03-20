# src/model.py

import torch.nn as nn
import torchvision.models as models


class MultiTaskResNet18(nn.Module):
    def __init__(self, num_conditions, num_classes=3):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_conditions * num_classes)

        self.num_conditions = num_conditions
        self.num_classes = num_classes

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(-1, self.num_conditions, self.num_classes)
        return out