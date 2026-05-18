from dataset import FashionDataset
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from torchvision.models import ResNet152_Weights
from model import ImageClassifier
import os

def main():
    """
    Train routine for fine tuning the ResNet152 with the new data
    """

    image_transform = ResNet152_Weights.IMAGENET1K_V2.transforms()
    g = torch.Generator()
    g.manual_seed(42)

    metadata_path = "../data/filter_data.csv"
    images_path = "../data/images/"

    df = pd.read_csv(metadata_path)

    classes = sorted(df["label"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    target_transform = lambda x: class_to_idx[x]

    dataset = FashionDataset(metadata_path, images_path, image_transform, target_transform)

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

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = ImageClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    size = len(train_loader.dataset)

    epochs = 200
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = "models/best_model.pth"

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))

        print(f"Loaded checkpoint from epoch {start_epoch}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # Training
        model.train()

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss_val = loss.item()
                current = (batch + 1) * len(X)
                print(f"loss: {loss_val:>7f} [{current:>5d}/{size:>5d}]")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)

        print(f"Validation Accuracy: {(100 * accuracy):.2f}%")
        print(f"Validation Loss: {val_loss:.6f}")

        # Save only if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, save_path)

            print("Model improved -> saved")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final test
    model.eval()
    correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_accuracy = correct / len(test_loader.dataset)

    print(f"\nFinal Test Accuracy: {(100 * test_accuracy):.2f}%")

if __name__ == "__main__":
    main()
