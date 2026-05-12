import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image

class FashioDataset(Dataset):
    def __init__(self, annotations_file : str, img_dir : str, transform=None, target_transform=None) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.loc[idx, "image"]))
        image = decode_image(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label