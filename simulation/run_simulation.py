"""In-process Flower simulation for FEM-NET — no network required.

Uses real processed CSVs from data/processed/<condition>.csv.
Splits each dataset into 3 non-IID hospital partitions before simulation.
"""

from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flwr as fl
import numpy as np
import pandas as pd
import torch

from federated.client import FemNetClient
from federated.strategy import KrumStrategy
from models.base_model import ClinicalMLP, get_parameters, set_parameters
from models.condition_models import save_model

RESULTS_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEATURES_DIM = 10
NUM_HOSPITALS = 3

FEATURES = [
    "age", "bmi", "fsh_level", "lh_level", "amh_level",
    "tsh_level", "cycle_length_days", "follicle_count",
    "fatigue_score", "weight_gain_kg",
]


def _split_non_iid(condition: str) -> list[str]:
    """
    Split real processed CSV into NUM_HOSPITALS non-IID partitions.
    Non-IID: sort by label then interleave to skew each hospital's class balance.
    Returns paths to temp partition CSVs.
    """
    src = os.path.join(PROCESSED_DIR, f"{condition}.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Processed dataset not found: {src}\n"
            f"Run data/preprocess_{condition}.py first (Suraj/Kuber task)."
        )

    df = pd.read_csv(src)
    df = df[FEATURES + ["label"]].dropna().reset_index(drop=True)

    # sort by label so positives cluster — split creates natural non-IID skew
    df_pos = df[df["label"] == 1].reset_index(drop=True)
    df_neg = df[df["label"] == 0].reset_index(drop=True)

    splits = []
    for i in range(NUM_HOSPITALS):
        # hospital i gets proportionally more positives or negatives
        pos_frac = [0.6, 0.5, 0.4][i]
        n = min(len(df) // NUM_HOSPITALS, len(df_pos), len(df_neg))
        n_pos = int(n * pos_frac)
        n_neg = n - n_pos

        pos_start = (i * len(df_pos) // NUM_HOSPITALS)
        neg_start = (i * len(df_neg) // NUM_HOSPITALS)

        part = pd.concat([
            df_pos.iloc[pos_start: pos_start + n_pos],
            df_neg.iloc[neg_start: neg_start + n_neg],
        ]).sample(frac=1, random_state=42 + i).reset_index(drop=True)

        part_path = os.path.join(PROCESSED_DIR, f"_sim_{condition}_h{i}.csv")
        part.to_csv(part_path, index=False)
        splits.append(part_path)

    return splits


def run_federated_simulation(condition: str = "pcos", num_rounds: int = 5) -> dict:
    print(f"\n[FEM-NET] Starting FL simulation: condition={condition}, rounds={num_rounds}")

    # split real data into hospital partitions
    hospital_csvs = _split_non_iid(condition)

    model = ClinicalMLP(input_dim=FEATURES_DIM)
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(model))

    strategy = KrumStrategy(
        num_byzantine=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_params,
    )

    def client_fn(cid: str) -> fl.client.NumPyClient:
        idx = int(cid) % NUM_HOSPITALS
        return FemNetClient(
            hospital_id=f"hospital_{idx}",
            condition=condition,
            data_path=hospital_csvs[idx],
        )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_HOSPITALS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # Flower History: metrics_distributed = {metric_name: [(round, value), ...]}
    losses = history.losses_distributed or []
    auc_rounds = history.metrics_distributed.get("auc", []) if history.metrics_distributed else []
    last_auc = float(auc_rounds[-1][1]) if auc_rounds else 0.0

    # save actual federated aggregated weights from strategy
    if strategy.last_aggregated_parameters is not None:
        final_ndarrays = fl.common.parameters_to_ndarrays(strategy.last_aggregated_parameters)
        global_model = ClinicalMLP(input_dim=FEATURES_DIM)
        set_parameters(global_model, final_ndarrays)
        save_model(global_model, condition)
        print(f"[FEM-NET] Global model saved → models/weights/{condition}_global.pt")
    else:
        print("[FEM-NET] Warning: no aggregated parameters found in strategy.")

    # clean up temp partition files
    for path in hospital_csvs:
        if os.path.exists(path):
            os.remove(path)

    summary = {
        "condition": condition,
        "num_rounds_completed": num_rounds,
        "participating_hospitals": NUM_HOSPITALS,
        "last_global_auc": last_auc,
        "losses_per_round": [(int(r), float(v)) for r, v in losses],
        "auc_per_round": [(int(r), float(v)) for r, v in auc_rounds],
    }

    out_path = os.path.join(RESULTS_DIR, f"results_{condition}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Simulation complete: {condition} ===")
    print(f"  Rounds : {num_rounds}")
    print(f"  Last AUC: {last_auc:.4f}")
    print(f"  Results → {out_path}")
    return summary


if __name__ == "__main__":
    ALL_CONDITIONS = ["pcos", "breast_cancer", "cervical", "thyroid", "diabetes", "cardiovascular", "heart"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="pcos", choices=ALL_CONDITIONS)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    run_federated_simulation(args.condition, args.rounds)
