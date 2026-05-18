import torchvision
from torch import nn, Tensor
from PIL import Image

class ImageClassifier(nn.Module):
    """
    Image classification model based on a pretrained ResNet152.

    This model classifies fashion images into one of 10 categories:
    Dress, Hat, Longsleeve, Outwear, Pants, Shirt, Shoes, Shorts, Skirt, T-Shirt.

    Attributes:
        model (torch.nn.Module): ResNet152 model with modified final layer.
        transforms (callable): Image preprocessing transforms matching pretrained weights.
    """
    def __init__(self) -> None:
        """
        Initialize the ImageClassifier model.

        Loads a pretrained ResNet152 model and replaces the final fully connected layer
        to match the number of fashion categories.

        Also initializes image preprocessing transforms compatible with ImageNet weights.
        """

        super().__init__()

        classes = [
            'Dress', 
            'Hat', 
            'Longsleeve', 
            'Outwear', 
            'Pants', 
            'Shirt', 
            'Shoes', 
            'Shorts', 
            'Skirt', 
            'T-Shirt']

        num_classes = len(classes)
        model = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model
        self.transforms = torchvision.models.ResNet152_Weights.IMAGENET1K_V2.transforms()

    def forward(self, x : Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input batch of images.

        Returns:
            Tensor: Raw class logits for each category.
        """
        return self.model(x)
    
    def transform(self, x : Image.Image) -> Tensor:
        """
        Preprocess a PIL image and prepare it for model inference.

        Args:
            x (PIL.Image.Image): Input image.

        Returns:
            Tensor: Preprocessed image tensor with batch dimension added.
        """
        x_tensor = self.transforms(x)
        return x_tensor.unsqueeze(0)