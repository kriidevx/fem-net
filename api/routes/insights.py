"""POST /api/insights — generate RAG insight card."""

from __future__ import annotations
from fastapi import APIRouter
from api.schemas import InsightRequest, InsightCard
from rag.insight_generator import generate_insight_card

router = APIRouter()

FEATURES = [
    "age", "bmi", "fsh_level", "lh_level", "amh_level",
    "tsh_level", "cycle_length_days", "follicle_count",
    "fatigue_score", "weight_gain_kg",
]


@router.post("/insights", response_model=InsightCard)
async def insights(req: InsightRequest) -> InsightCard:
    patient_dict = {f: getattr(req.patient, f) for f in FEATURES}
    card = generate_insight_card(patient_dict, req.risk_score, req.patient.condition)
    return InsightCard(**card)
