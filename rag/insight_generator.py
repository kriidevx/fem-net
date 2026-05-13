"""RAG insight card generator using LangChain + GPT-4o-mini (with fallback)."""

from __future__ import annotations
import json
import os
from typing import Dict, List

from rag.embedder import EvidenceStore
from rag.pubmed_fetcher import fetch_pubmed_abstracts

RISK_THRESHOLDS = {"low": 0.35, "moderate": 0.65}


def _risk_level(score: float) -> str:
    if score < RISK_THRESHOLDS["low"]:
        return "Low"
    if score < RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    return "High"


def _feature_summary(features: Dict) -> str:
    parts = [f"{k.replace('_', ' ')}: {v:.2f}" if isinstance(v, float) else f"{k.replace('_', ' ')}: {v}"
             for k, v in features.items()]
    return "; ".join(parts)


def _ensure_store(condition: str) -> EvidenceStore:
    store = EvidenceStore(condition)
    if not store.is_built():
        docs = fetch_pubmed_abstracts(condition)
        store.build(docs)
    return store


def generate_insight_card(
    patient_features: Dict,
    risk_score: float,
    condition: str,
) -> Dict:
    level = _risk_level(risk_score)
    try:
        store = _ensure_store(condition)
    except Exception as exc:
        print(f"[RAG] Evidence store init failed: {exc}. Using fallback mode.")
        store = None
    summary_text = _feature_summary(patient_features)
    query = f"{condition} risk factors: {summary_text}"
    try:
        retrieved = store.query(query, k=3) if store is not None else []
    except Exception as exc:
        print(f"[RAG] Evidence query failed: {exc}. Using fallback mode.")
        retrieved = []

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        return _llm_card(patient_features, risk_score, level, condition, retrieved, api_key)
    return _fallback_card(patient_features, risk_score, level, condition, retrieved)


def _llm_card(features: Dict, score: float, level: str, condition: str, evidence: List[str], api_key: str) -> Dict:
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0.2)
        evidence_block = "\n---\n".join(evidence[:3]) or "No evidence retrieved."
        feature_str = _feature_summary(features)

        prompt = ChatPromptTemplate.from_template(
            "You are a clinical AI assistant specialized in women's health early detection.\n"
            "Patient features: {features}\n"
            "Condition assessed: {condition}\n"
            "Risk score: {score:.2f} ({level} risk)\n\n"
            "Retrieved evidence:\n{evidence}\n\n"
            "Generate a structured Early Risk Insight Card as JSON with exactly these keys:\n"
            "- risk_level (string)\n"
            "- key_indicators (list of 3-5 strings, citing specific patient values)\n"
            "- evidence_summary (string, 2-3 sentences citing retrieved titles)\n"
            "- recommended_next_steps (list of 3-5 strings)\n\n"
            "Return ONLY valid JSON, no markdown."
        )
        chain = prompt | llm
        result = chain.invoke({
            "features": feature_str,
            "condition": condition,
            "score": score,
            "level": level,
            "evidence": evidence_block,
        })
        return json.loads(result.content)
    except Exception as exc:
        print(f"[RAG] LLM call failed: {exc}. Falling back.")
        return _fallback_card(features, score, level, condition, evidence)


def _fallback_card(features: Dict, score: float, level: str, condition: str, evidence: List[str]) -> Dict:
    indicators = [
        f"{k.replace('_', ' ').title()}: {v:.2f}" if isinstance(v, float) else f"{k.replace('_', ' ').title()}: {v}"
        for k, v in list(features.items())[:5]
    ]
    evidence_titles = []
    for e in evidence:
        first_sentence = e.split(".")[0]
        evidence_titles.append(first_sentence[:120])

    return {
        "risk_level": level,
        "key_indicators": indicators,
        "evidence_summary": (
            f"Based on retrieved literature: {'; '.join(evidence_titles[:2])}."
            if evidence_titles else f"Risk score {score:.2f} indicates {level.lower()} risk for {condition}."
        ),
        "recommended_next_steps": [
            f"Consult a specialist for {condition} evaluation",
            "Schedule comprehensive hormone panel blood work",
            "Track symptom progression with a health diary",
            "Review family history for related conditions",
            "Follow up in 4-6 weeks with primary care physician",
        ],
    }
