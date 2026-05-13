"""Shared constants used across FEM-NET modules.

Keep these centralized to avoid feature-order/condition mismatches between:
- training
- federated simulation
- inference (predict)
- insights
"""

from __future__ import annotations

# 10-feature vector (order matters)
FEATURES: list[str] = [
    "age",
    "bmi",
    "fsh_level",
    "lh_level",
    "amh_level",
    "tsh_level",
    "cycle_length_days",
    "follicle_count",
    "fatigue_score",
    "weight_gain_kg",
]

# Supported demo conditions (must match models/condition_models.py)
SUPPORTED_CONDITIONS: list[str] = [
    "pcos",
    "breast_cancer",
    "cervical",
    "thyroid",
    "diabetes",
    "cardiovascular",
    "heart",
]
