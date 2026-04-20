from __future__ import annotations

from collections import defaultdict

import numpy as np

from evacuation_vug.network import TreeNetwork
from evacuation_vug.simulator import SimulationResult
from evacuation_vug.surrogate import tokens_from_schedule, route_service_counts, utilities_by_intersection, welfare


def completion_percentage(result: SimulationResult) -> float:
    total = len(result.completed) + len(result.unfinished)
    return len(result.completed) / total if total else 1.0


def travel_time_by_generator(result: SimulationResult) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for v in result.completed:
        if v.completion_time is not None:
            out[v.origin].append(v.completion_time - v.birth_t)
    return dict(out)


def queue_wait_summary(result: SimulationResult) -> dict[str, dict[str, float]]:
    out = {}
    for i, waits in result.queue_wait_stats.items():
        arr = np.array(waits, dtype=float)
        out[i] = {"mean_wait": float(arr.mean()) if arr.size else 0.0, "max_wait": float(arr.max()) if arr.size else 0.0}
    return out


def policy_summary(network: TreeNetwork, result: SimulationResult, surrogate_cfg: dict) -> dict:
    tokens = tokens_from_schedule(result.schedule)
    travel = travel_time_by_generator(result)
    return {
        "policy": result.policy,
        "completion_pct": completion_percentage(result),
        "mean_travel_time": float(np.mean([t for ts in travel.values() for t in ts])) if travel else 0.0,
        "route_service_counts": route_service_counts(network, tokens),
        "surrogate_welfare": welfare(
            network,
            tokens,
            surrogate_cfg["alpha"],
            surrogate_cfg["phi_type"],
            surrogate_cfg["phi_params"],
        ),
        "player_utilities": utilities_by_intersection(
            network,
            tokens,
            surrogate_cfg["alpha"],
            surrogate_cfg["phi_type"],
            surrogate_cfg["phi_params"],
        ),
        "generator_travel": {
            g: {
                "mean": float(np.mean(ts)) if ts else 0.0,
                "max": float(np.max(ts)) if ts else 0.0,
                "count": len(ts),
                "raw": list(ts),
            }
            for g, ts in travel.items()
        },
        "intersection_wait": queue_wait_summary(result),
    }
