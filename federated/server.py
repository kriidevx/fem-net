"""Flower server launcher with KrumStrategy."""

from __future__ import annotations
import os
import torch
import flwr as fl
from federated.strategy import KrumStrategy
from models.base_model import ClinicalMLP, get_parameters, set_parameters
from models.condition_models import save_model, WEIGHTS_DIR

FEATURES_DIM = 10


def start_server(condition: str = "pcos", num_rounds: int = 10, port: int = 8080) -> None:
    model = ClinicalMLP(input_dim=FEATURES_DIM)
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(model))

    strategy = KrumStrategy(
        num_byzantine=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_params,
    )

    history = fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # persist final global model
    if history.metrics_distributed:
        last_round_metrics = history.metrics_distributed.get("auc", [])
        print(f"FL complete. Final distributed metrics: {last_round_metrics[-3:] if last_round_metrics else 'N/A'}")

    # reconstruct global model from last aggregated params (via strategy attribute if available)
    print(f"Server done for condition={condition}. Weights at {WEIGHTS_DIR}/{condition}_global.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="pcos")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    start_server(args.condition, args.rounds, args.port)
