# src/train.py

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

from dataset import SagT2Dataset
from model import MultiTaskResNet18
from evaluate import evaluate


IMG_ROOT = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"


def main():

    # Load dataframe (you can adapt path later)
    df = pd.read_csv("train_processed.csv")  # save your cleaned df

    id_cols = ["study_id", "series_id_sag_t2"]
    label_cols = [c for c in df.columns if c not in id_cols]

    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = SagT2Dataset(train_df, label_cols, IMG_ROOT, transform)
    val_dataset = SagT2Dataset(val_df, label_cols, IMG_ROOT, transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskResNet18(num_conditions=len(label_cols)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            B, C, K = outputs.shape
            logits = outputs.view(B * C, K)
            target = labels.view(B * C)

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")


if __name__ == "__main__":
    main()