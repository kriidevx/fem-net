"""Factory for condition-specific ClinicalMLP instances with weight persistence."""

from __future__ import annotations
import os
import torch
from models.base_model import ClinicalMLP

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
INPUT_DIM = 10  # all conditions share same 10-feature vector

SUPPORTED = {"pcos", "breast_cancer", "cervical", "thyroid", "diabetes", "cardiovascular", "heart"}


def get_model(condition: str) -> ClinicalMLP:
    if condition not in SUPPORTED:
        raise ValueError(f"Unknown condition '{condition}'. Supported: {SUPPORTED}")
    model = ClinicalMLP(input_dim=INPUT_DIM)
    weight_path = _weight_path(condition)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    return model


def save_model(model: ClinicalMLP, condition: str) -> str:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = _weight_path(condition)
    torch.save(model.state_dict(), path)
    return path


def _weight_path(condition: str) -> str:
    return os.path.join(WEIGHTS_DIR, f"{condition}_global.pt")
