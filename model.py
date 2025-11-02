# CNN Model Spec for Otolith Shape Classification
import torch
import torch.nn as nn
import torchvision.models as models

class OtolithResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
    def forward(self, x):
        return self.base(x)

# Input size: 224x224, RGB
# Preprocessing: Resize, Normalize (ImageNet mean/std)
# Grad-CAM: see gradcam.py
