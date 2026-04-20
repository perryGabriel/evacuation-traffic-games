from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from evacuation_vug.network import TreeNetwork
from evacuation_vug.policies import Policy, PolicyContext
from evacuation_vug.queues import EdgeQueue
from evacuation_vug.vehicles import Vehicle


@dataclass(slots=True)
class SimulationResult:
    policy: str
    horizon: int
    seed: int
    completed: list[Vehicle]
    unfinished: list[Vehicle]
    schedule: dict[int, dict[str, str | None]]
    generated_by_origin: dict[str, int]
    queue_wait_stats: dict[str, list[int]]

    def to_json(self, path: str | Path) -> None:
        payload = {
            "policy": self.policy,
            "horizon": self.horizon,
            "seed": self.seed,
            "generated_by_origin": self.generated_by_origin,
            "completed_count": len(self.completed),
            "unfinished_count": len(self.unfinished),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(slots=True)
class Simulator:
    network: TreeNetwork
    generation_rates: dict[str, float]
    seed: int
    capture_snapshots: bool = False

    queues: dict[tuple[str, str], EdgeQueue] = field(init=False)

    def __post_init__(self) -> None:
        self.queues = {}
        for u, v, data in self.network.graph.edges(data=True):
            self.queues[(u, v)] = EdgeQueue(
                edge=(u, v),
                capacity=int(data["capacity"]),
                hold_base=float(data["hold_base"]),
                hold_log_coeff=float(data["hold_log_coeff"]),
            )

    def run(self, horizon: int, policy: Policy) -> SimulationResult:
        rng = np.random.default_rng(self.seed)
        vehicle_id = 0
        completed: list[Vehicle] = []
        unfinished: dict[int, Vehicle] = {}
        schedule: dict[int, dict[str, str | None]] = {}
        generated = {g: 0 for g in self.network.generators}
        queue_wait_stats: dict[str, list[int]] = {i: [] for i in self.network.intersections}

        for t in range(horizon):
            for v in unfinished.values():
                v.tick()

            # Generate
            for g in self.network.generators:
                if rng.random() < self.generation_rates[g]:
                    route = self.network.route_to_safe(g)
                    car = Vehicle(vehicle_id=vehicle_id, origin=g, birth_t=t, route=route)
                    vehicle_id += 1
                    first = (route[0], route[1])
                    if self.queues[first].enqueue(car):
                        unfinished[car.vehicle_id] = car
                        generated[g] += 1

            # Policy context
            intersection_to_branches = {
                i: [self.network.branch_id(i, u) for u, _ in self.network.graph.in_edges(i)]
                for i in self.network.intersections
            }
            q_waits: dict[tuple[str, str], int] = {}
            q_cum: dict[tuple[str, str], int] = {}
            for i in self.network.intersections:
                for u, _ in self.network.graph.in_edges(i):
                    b = self.network.branch_id(i, u)
                    eq = self.queues[(u, i)]
                    q_waits[(i, b)] = eq.current_wait()
                    q_cum[(i, b)] = eq.cumulative_wait()
                    queue_wait_stats[i].append(eq.current_wait())

            ctx = PolicyContext(t, intersection_to_branches, q_waits, q_cum)
            actions = policy.choose(ctx)
            schedule[t] = actions

            # Serve one vehicle at each intersection on chosen branch if eligible
            moved: list[tuple[Vehicle, tuple[str, str]]] = []
            for i, branch in actions.items():
                if branch is None:
                    continue
                upstream = branch.split("->")[0]
                q = self.queues[(upstream, i)]
                v = q.pop_eligible()
                if v is None:
                    continue
                if i == self.network.safe_node:
                    v.completion_time = t
                    completed.append(v)
                    unfinished.pop(v.vehicle_id, None)
                    continue
                next_index = v.route.index(i) + 1
                if next_index >= len(v.route):
                    v.completion_time = t
                    completed.append(v)
                    unfinished.pop(v.vehicle_id, None)
                    continue
                nxt = v.route[next_index]
                if nxt == self.network.safe_node:
                    v.completion_time = t
                    completed.append(v)
                    unfinished.pop(v.vehicle_id, None)
                else:
                    moved.append((v, (i, nxt)))

            # Complete moves if destination queue has room
            for v, edge in moved:
                if self.queues[edge].enqueue(v):
                    continue
                # if blocked, reinsert into prior queue front (fallback)
                prev = v.route[v.route.index(edge[0]) - 1]
                self.queues[(prev, edge[0])].vehicles.appendleft(v)

        return SimulationResult(policy.name, horizon, self.seed, completed, list(unfinished.values()), schedule, generated, queue_wait_stats)
