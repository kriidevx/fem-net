"""POST /api/insights — generate RAG insight card."""

from __future__ import annotations
from fastapi import APIRouter
from api.schemas import InsightRequest, InsightCard
from rag.insight_generator import generate_insight_card
from core.constants import FEATURES

router = APIRouter()


@router.post("/insights", response_model=InsightCard)
async def insights(req: InsightRequest) -> InsightCard:
    patient_dict = {f: getattr(req.patient, f) for f in FEATURES}
    card = generate_insight_card(patient_dict, req.risk_score, req.patient.condition)
    return InsightCard(**card)
