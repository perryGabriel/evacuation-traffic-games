from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from evacuation_vug.game import (
    feasible_actions_for_intersection,
    profile_welfare,
    unilateral_deviation_profiles,
)
from evacuation_vug.network import TreeNetwork


@dataclass(slots=True)
class ExactGameResult:
    equilibria: list[dict[str, tuple[str, ...]]]
    optimal_welfare: float
    best_equilibrium_welfare: float
    poa: float


def enumerate_pure_nash(network: TreeNetwork, horizon: int, surrogate_cfg: dict) -> ExactGameResult:
    action_spaces = {}
    for i in network.intersections:
        branches = [network.branch_id(i, u) for u, _ in network.graph.in_edges(i)]
        action_spaces[i] = feasible_actions_for_intersection(branches, horizon)

    players = sorted(network.intersections)
    all_profiles = []
    for combo in product(*(action_spaces[p] for p in players)):
        all_profiles.append({p: a for p, a in zip(players, combo)})

    welfare_cache: dict[str, float] = {}

    def w(profile: dict[str, tuple[str, ...]]) -> float:
        key = repr(profile)
        if key not in welfare_cache:
            welfare_cache[key] = profile_welfare(network, horizon, profile, surrogate_cfg)
        return welfare_cache[key]

    equilibria = []
    for p in all_profiles:
        is_ne = True
        for i in players:
            base = w(p)
            for dev in unilateral_deviation_profiles(network, horizon, p, i):
                if w(dev) > base + 1e-9:
                    is_ne = False
                    break
            if not is_ne:
                break
        if is_ne:
            equilibria.append(p)

    optimal = max(w(p) for p in all_profiles) if all_profiles else 0.0
    best_eq = max((w(p) for p in equilibria), default=0.0)
    poa = best_eq / optimal if optimal > 0 else 1.0
    return ExactGameResult(equilibria, optimal, best_eq, poa)


def best_response_dynamics(
    network: TreeNetwork,
    horizon: int,
    surrogate_cfg: dict,
    seed: int,
    restarts: int = 5,
    max_iters: int = 50,
) -> dict:
    rng = random.Random(seed)
    players = sorted(network.intersections)
    action_spaces = {
        i: feasible_actions_for_intersection([network.branch_id(i, u) for u, _ in network.graph.in_edges(i)], horizon)
        for i in players
    }

    best_profile = None
    best_w = -1.0

    for _ in range(restarts):
        profile = {i: rng.choice(action_spaces[i]) for i in players}
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in players:
                current_w = profile_welfare(network, horizon, profile, surrogate_cfg)
                best_i = profile[i]
                best_i_w = current_w
                for a in action_spaces[i]:
                    trial = dict(profile)
                    trial[i] = a
                    trial_w = profile_welfare(network, horizon, trial, surrogate_cfg)
                    if trial_w > best_i_w + 1e-9:
                        best_i, best_i_w = a, trial_w
                if best_i != profile[i]:
                    profile[i] = best_i
                    improved = True
        final_w = profile_welfare(network, horizon, profile, surrogate_cfg)
        if final_w > best_w:
            best_w = final_w
            best_profile = dict(profile)

    return {"approx_best_profile": best_profile, "approx_welfare": best_w, "note": "Approximate best-response dynamics"}
