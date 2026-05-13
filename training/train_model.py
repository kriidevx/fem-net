"""
FEM-NET unified trainer.
Reads: data/processed/<condition>.csv
Writes: models/weights/<condition>_global.pt
        models/weights/<condition>_norm_stats.json
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

from models.base_model import ClinicalMLP
from core.constants import FEATURES, SUPPORTED_CONDITIONS


def train(condition: str, epochs: int = 50, lr: float = 3e-4, batch: int = 64):
    csv_path = f"data/processed/{condition}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate schema
    missing_cols = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    X = df[FEATURES].values.astype("float32")
    y = df["label"].values.astype("float32")

    # Compute and save norm stats BEFORE normalising
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8

    os.makedirs("models/weights", exist_ok=True)
    norm_stats = {f: {"mean": float(mean[i]), "std": float(std[i])} for i, f in enumerate(FEATURES)}
    stats_path = f"models/weights/{condition}_norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Norm stats saved → {stats_path}")

    X = (X - mean) / std

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Positive rate: {y.mean():.1%}")

    loader_train = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch, shuffle=True
    )
    loader_val = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch
    )

    model   = ClinicalMLP(input_dim=len(FEATURES))
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_auc   = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader_train:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader_val:
                preds.extend(model(xb).numpy())
                labels.extend(yb.numpy())

        auc = roc_auc_score(labels, preds)
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}  Val AUC: {auc:.4f}  Best: {best_auc:.4f}")

    # Save best checkpoint
    weights_path = f"models/weights/{condition}_global.pt"
    torch.save(best_state, weights_path)

    # Final confusion matrix
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_preds = (model(torch.from_numpy(X_val)).numpy() >= 0.5).astype(int)
    cm = confusion_matrix(y_val.astype(int), final_preds)

    print(f"\n{'='*40}")
    print(f"CONDITION : {condition.upper()}")
    print(f"Best AUC  : {best_auc:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(f"Weights   → {weights_path}")
    print(f"Norm stats→ {stats_path}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=SUPPORTED_CONDITIONS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    train(args.condition, args.epochs, args.lr, args.batch)