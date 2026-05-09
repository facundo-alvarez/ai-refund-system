import os
import torchvision
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        classes = [
            'Electronics', 
            'Apparel', 
            'Home & Kitchen', 
            'Beauty', 
            'Toys', 
            'Sports', 
            'Books', 
            'Automotive', 
            'Groceries', 
            'Pet Supplies']

        num_classes = len(classes)
        model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model
        self.flatten = nn.Flatten()
        self.transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()

    def forward(self, x : Tensor) -> Tensor:
        return self.model.forward(x)
    
    def transform(self, x : Image.Image) -> Tensor:
        x_tensor = self.transforms(x)
        return x_tensor.unsqueeze(0)