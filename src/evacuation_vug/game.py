from __future__ import annotations

from itertools import product

from evacuation_vug.network import TreeNetwork
from evacuation_vug.surrogate import welfare
from evacuation_vug.types import Token


def feasible_actions_for_intersection(branches: list[str], horizon: int) -> list[tuple[str, ...]]:
    if not branches:
        return [tuple()]
    return list(product(branches, repeat=horizon))


def schedule_from_profile(network: TreeNetwork, horizon: int, profile: dict[str, tuple[str, ...]]) -> set[Token]:
    tokens: set[Token] = set()
    for i in network.intersections:
        seq = profile[i]
        for t, b in enumerate(seq):
            tokens.add((i, b, t))
    return tokens


def unilateral_deviation_profiles(
    network: TreeNetwork,
    horizon: int,
    profile: dict[str, tuple[str, ...]],
    deviator: str,
) -> list[dict[str, tuple[str, ...]]]:
    branches = [network.branch_id(deviator, u) for u, _ in network.graph.in_edges(deviator)]
    deviations = []
    for action in feasible_actions_for_intersection(branches, horizon):
        p2 = dict(profile)
        p2[deviator] = action
        deviations.append(p2)
    return deviations


def profile_welfare(network: TreeNetwork, horizon: int, profile: dict[str, tuple[str, ...]], surrogate_cfg: dict) -> float:
    toks = schedule_from_profile(network, horizon, profile)
    return welfare(network, toks, surrogate_cfg["alpha"], surrogate_cfg["phi_type"], surrogate_cfg["phi_params"])
