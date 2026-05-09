"""Pydantic request/response models for FEM-NET API."""

from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    age: float = Field(..., ge=18, le=45)
    bmi: float = Field(..., ge=10, le=60)
    fsh_level: float = Field(..., ge=0)
    lh_level: float = Field(..., ge=0)
    amh_level: float = Field(..., ge=0)
    tsh_level: float = Field(..., ge=0)
    cycle_length_days: float = Field(..., ge=1)
    follicle_count: float = Field(..., ge=0)
    fatigue_score: float = Field(..., ge=1, le=10)
    weight_gain_kg: float = Field(..., ge=0)
    condition: Literal["pcos", "breast_cancer", "cervical", "thyroid", "diabetes", "cardiovascular", "heart"] = "pcos"


class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    model_version: str


class InsightRequest(BaseModel):
    patient: PatientInput
    risk_score: float = Field(..., ge=0.0, le=1.0)


class InsightCard(BaseModel):
    risk_level: str
    key_indicators: List[str]
    evidence_summary: str
    recommended_next_steps: List[str]


class FederatedStatus(BaseModel):
    condition: str
    num_rounds_completed: int
    participating_hospitals: int
    last_global_auc: float
