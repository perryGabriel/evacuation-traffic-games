from __future__ import annotations

from collections import defaultdict

from evacuation_vug.network import TreeNetwork
from evacuation_vug.types import Token


def phi_capped(n: int, d: int) -> float:
    return float(min(n, d))


def phi_geometric(n: int, beta: float) -> float:
    if n <= 0:
        return 0.0
    if beta == 1.0:
        return float(n)
    return float(sum(beta ** (m - 1) for m in range(1, n + 1)))


def route_service_counts(network: TreeNetwork, tokens: set[Token]) -> dict[str, int]:
    by_gen = {g: 0 for g in network.generators}
    for g in network.generators:
        route_branches = network.route_branch_for_generator(g)
        count = 0
        for i, b in route_branches.items():
            for ii, bb, _t in tokens:
                if ii == i and bb == b:
                    count += 1
        by_gen[g] = count
    return by_gen


def welfare(
    network: TreeNetwork,
    tokens: set[Token],
    alpha: dict[str, float],
    phi_type: dict[str, str],
    phi_params: dict[str, dict[str, float]],
) -> float:
    n_k = route_service_counts(network, tokens)
    total = 0.0
    for g, n in n_k.items():
        if phi_type[g] == "capped":
            value = phi_capped(n, int(phi_params[g]["D"]))
        elif phi_type[g] == "geometric":
            value = phi_geometric(n, float(phi_params[g]["beta"]))
        else:
            raise ValueError(f"Unknown phi type: {phi_type[g]}")
        total += alpha[g] * value
    return total


def utilities_by_intersection(
    network: TreeNetwork,
    tokens: set[Token],
    alpha: dict[str, float],
    phi_type: dict[str, str],
    phi_params: dict[str, dict[str, float]],
) -> dict[str, float]:
    w = welfare(network, tokens, alpha, phi_type, phi_params)
    out: dict[str, float] = {}
    for i in network.intersections:
        s_minus_i = {tok for tok in tokens if tok[0] != i}
        out[i] = w - welfare(network, s_minus_i, alpha, phi_type, phi_params)
    return out


def tokens_from_schedule(schedule: dict[int, dict[str, str | None]]) -> set[Token]:
    tokens: set[Token] = set()
    for t, actions in schedule.items():
        for i, b in actions.items():
            if b is not None:
                tokens.add((i, b, t))
    return tokens


def tokens_grouped_by_intersection(tokens: set[Token]) -> dict[str, set[Token]]:
    grouped: dict[str, set[Token]] = defaultdict(set)
    for tok in tokens:
        grouped[tok[0]].add(tok)
    return dict(grouped)
