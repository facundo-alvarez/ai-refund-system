from dataset import FashioDataset
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch
import numpy as np
import pandas as pd
from torchvision.models import ResNet50_Weights

image_transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
g = torch.Generator()
g.manual_seed(42)

metadata_path = "../data/filter_data.csv"
images_path = "../data/images/"

df = pd.read_csv(metadata_path)

classes = sorted(df["label"].unique())
class_to_idx = {cls: i for i, cls in enumerate(classes)}
target_transform = lambda x: class_to_idx[x]

dataset = FashioDataset(metadata_path, images_path, image_transform, target_transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(dataset,[train_size, val_size, test_size],generator=g)

train_indices = train_data.indices
train_labels = [class_to_idx[dataset.img_labels.iloc[i]["label"]] for i in train_indices]
train_labels = np.array(train_labels)

class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]

train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True, generator=g)

train_loader = DataLoader(train_data, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
