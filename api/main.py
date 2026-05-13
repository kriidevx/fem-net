"""FEM-NET FastAPI application."""

from __future__ import annotations
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.routes import predict, insights, federated
from core.constants import SUPPORTED_CONDITIONS

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")


ALL_CONDITIONS = SUPPORTED_CONDITIONS
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "weights")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # check real datasets exist — never generate synthetic data
    missing = [c for c in ALL_CONDITIONS if not os.path.exists(os.path.join(PROCESSED_DIR, f"{c}.csv"))]
    if missing:
        print(f"[WARN] Missing processed CSVs for: {missing}")
        print("[WARN] Some training/simulation flows may fail. Run data/preprocess_<condition>.py.")

    # warn if model artifacts are missing (non-fatal for submission)
    for cond in ALL_CONDITIONS:
        weights_path = os.path.join(WEIGHTS_DIR, f"{cond}_global.pt")
        stats_path = os.path.join(WEIGHTS_DIR, f"{cond}_norm_stats.json")
        if not os.path.exists(weights_path):
            print(f"[WARN] Missing weights for {cond} model ({weights_path}). Predictions may be untrained.")
        if not os.path.exists(stats_path):
            print(f"[WARN] Missing norm stats for {cond} model ({stats_path}). Predictions run unnormalised.")

    # pre-warm FAISS indexes (best-effort, requires network for PubMed)
    try:
        from rag.embedder import EvidenceStore
        from rag.pubmed_fetcher import fetch_pubmed_abstracts
        for cond in ALL_CONDITIONS:
            store = EvidenceStore(cond)
            if not store.is_built():
                print(f"[startup] Building FAISS index for {cond}...")
                docs = fetch_pubmed_abstracts(cond)
                store.build(docs)
    except Exception as exc:
        print(f"[startup] RAG pre-warm skipped: {exc}")

    yield


app = FastAPI(
    title="FEM-NET API",
    description="Federated Early Misdiagnosis Network — privacy-preserving women's health AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")
app.include_router(insights.router, prefix="/api")
app.include_router(federated.router, prefix="/api")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "FEM-NET"}


# serve dashboard at root
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
