"""GET /api/federated/status, POST /api/federated/run."""

from __future__ import annotations
import json
import os
from fastapi import APIRouter, BackgroundTasks, Query
from api.schemas import FederatedStatus

router = APIRouter()

SIMULATION_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "simulation")


def _results_path(condition: str) -> str:
    return os.path.join(SIMULATION_DIR, f"results_{condition}.json")


@router.get("/federated/status", response_model=FederatedStatus)
async def federated_status(condition: str = Query("pcos")) -> FederatedStatus:
    path = _results_path(condition)
    if not os.path.exists(path):
        return FederatedStatus(
            condition=condition,
            num_rounds_completed=0,
            participating_hospitals=0,
            last_global_auc=0.0,
        )
    with open(path) as f:
        data = json.load(f)
    return FederatedStatus(
        condition=data.get("condition", condition),
        num_rounds_completed=data.get("num_rounds_completed", 0),
        participating_hospitals=data.get("participating_hospitals", 0),
        last_global_auc=data.get("last_global_auc", 0.0),
    )


def _run_simulation(condition: str, rounds: int) -> None:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from simulation.run_simulation import run_federated_simulation
    run_federated_simulation(condition=condition, num_rounds=rounds)


@router.post("/federated/run")
async def federated_run(
    background_tasks: BackgroundTasks,
    condition: str = Query("pcos"),
    rounds: int = Query(5, ge=1, le=20),
) -> dict:
    background_tasks.add_task(_run_simulation, condition, rounds)
    return {"status": "started", "condition": condition, "rounds": rounds}
