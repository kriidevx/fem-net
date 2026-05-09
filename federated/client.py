"""Flower NumPy client — trains local ClinicalMLP on one hospital's CSV."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple

import flwr as fl
from models.base_model import ClinicalMLP, get_parameters, set_parameters

FEATURES = [
    "age", "bmi", "fsh_level", "lh_level", "amh_level",
    "tsh_level", "cycle_length_days", "follicle_count",
    "fatigue_score", "weight_gain_kg",
]
BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LR = 3e-4


class FemNetClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: str, condition: str, data_path: str):
        self.hospital_id = hospital_id
        self.condition = condition
        self.device = torch.device("cpu")
        self.model = ClinicalMLP(input_dim=len(FEATURES)).to(self.device)

        df = pd.read_csv(data_path)
        X = df[FEATURES].values.astype(np.float32)
        y = df["label"].values.astype(np.float32)

        # per-hospital normalisation
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=BATCH_SIZE,
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        set_parameters(self.model, parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        criterion = nn.BCELoss()

        total_loss = 0.0
        total_samples = 0
        for _ in range(LOCAL_EPOCHS):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(X_batch)
                total_samples += len(X_batch)

        avg_loss = total_loss / max(total_samples, 1)
        return get_parameters(self.model), total_samples, {"train_loss": avg_loss, "num_samples": total_samples}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)
        self.model.eval()
        criterion = nn.BCELoss()

        all_preds, all_labels = [], []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
                total_samples += len(X_batch)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = float(np.mean((np.array(all_preds) >= 0.5) == np.array(all_labels)))
        try:
            auc = float(roc_auc_score(all_labels, all_preds))
        except Exception:
            auc = 0.5

        return avg_loss, total_samples, {"accuracy": accuracy, "auc": auc}
