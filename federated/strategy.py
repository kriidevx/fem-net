"""Byzantine-robust Multi-Krum aggregation strategy for Flower."""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, Union, Dict
from flwr.common import (
    FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _krum_scores(updates: List[List[np.ndarray]], n_to_keep: int) -> List[float]:
    """Compute Multi-Krum score for each client update (lower = more trustworthy)."""
    n = len(updates)
    flat = [np.concatenate([p.flatten() for p in u]) for u in updates]

    scores = []
    for i in range(n):
        dists = sorted(
            np.sum((flat[i] - flat[j]) ** 2) for j in range(n) if j != i
        )
        scores.append(sum(dists[: n - n_to_keep - 2]) if n > 2 else 0.0)
    return scores


class KrumStrategy(FedAvg):
    """FedAvg with Multi-Krum pre-filtering to drop Byzantine outliers."""

    def __init__(self, num_byzantine: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_byzantine = num_byzantine
        self.last_aggregated_parameters: Optional[Parameters] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        updates = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_clients = len(updates)

        if num_clients <= 2:
            aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        else:
            n_to_keep = max(1, num_clients - self.num_byzantine)
            scores = _krum_scores(updates, n_to_keep)
            selected_indices = sorted(range(num_clients), key=lambda i: scores[i])[:n_to_keep]
            selected_results = [results[i] for i in selected_indices]
            print(
                f"[Krum] round {server_round}: kept {len(selected_results)}/{num_clients} clients "
                f"(dropped {num_clients - len(selected_results)} outliers)"
            )
            aggregated, metrics = super().aggregate_fit(server_round, selected_results, failures)

        if aggregated is not None:
            self.last_aggregated_parameters = aggregated
        return aggregated, metrics
