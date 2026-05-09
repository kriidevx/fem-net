"""POST /api/predict — run ClinicalMLP inference."""

from __future__ import annotations
import json
import os
import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from api.schemas import PatientInput, PredictionResponse
from models.base_model import ClinicalMLP, set_parameters
from models.condition_models import get_model

router = APIRouter()

FEATURES = [
    "age", "bmi", "fsh_level", "lh_level", "amh_level",
    "tsh_level", "cycle_length_days", "follicle_count",
    "fatigue_score", "weight_gain_kg",
]
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "weights")

_norm_cache: dict = {}


def _load_norm_stats(condition: str) -> dict:
    """Load per-condition norm stats written by training/train_model.py."""
    global _norm_cache
    if condition in _norm_cache:
        return _norm_cache[condition]
    path = os.path.join(WEIGHTS_DIR, f"{condition}_norm_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            _norm_cache[condition] = json.load(f)
    else:
        _norm_cache[condition] = {}
    return _norm_cache[condition]


def _normalise(features: np.ndarray, condition: str) -> np.ndarray:
    stats = _load_norm_stats(condition)
    if not stats:
        return features  # no stats yet — return raw (untrained model state)
    means = np.array([stats[f]["mean"] if f in stats else 0.0 for f in FEATURES], dtype=np.float32)
    stds = np.array([max(stats[f]["std"], 1e-8) if f in stats else 1.0 for f in FEATURES], dtype=np.float32)
    return (features - means) / stds


def _risk_level(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.65:
        return "Moderate"
    return "High"


def _model_version(condition: str) -> str:
    weight_path = os.path.join(WEIGHTS_DIR, f"{condition}_global.pt")
    if os.path.exists(weight_path):
        mtime = int(os.path.getmtime(weight_path))
        return f"{condition}-v{mtime}"
    return f"{condition}-untrained"


@router.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput) -> PredictionResponse:
    condition = patient.condition
    model = get_model(condition)
    model.eval()

    raw = np.array([getattr(patient, f) for f in FEATURES], dtype=np.float32)
    normed = _normalise(raw, condition).astype(np.float32)

    with torch.no_grad():
        tensor = torch.from_numpy(normed).unsqueeze(0)
        score = float(model(tensor).item())

    return PredictionResponse(
        risk_score=round(score, 4),
        risk_level=_risk_level(score),
        model_version=_model_version(condition),
    )
