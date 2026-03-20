# src/evaluate.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch


def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            B, C, K = outputs.shape
            logits = outputs.view(B * C, K)
            target = labels.view(B * C)

            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return acc, f1