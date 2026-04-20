from __future__ import annotations

from dataclasses import dataclass, field

from evacuation_vug.types import PolicyAction


@dataclass(slots=True)
class PolicyContext:
    t: int
    intersection_to_branches: dict[str, list[str]]
    queue_waits: dict[tuple[str, str], int]
    queue_cumulative_waits: dict[tuple[str, str], int]


class Policy:
    name: str

    def choose(self, ctx: PolicyContext) -> PolicyAction:
        raise NotImplementedError


class AllOpenPolicy(Policy):
    """Selects lexicographically first branch at each intersection (local, deterministic)."""

    name = "all_open"

    def choose(self, ctx: PolicyContext) -> PolicyAction:
        return {i: sorted(branches)[0] if branches else None for i, branches in ctx.intersection_to_branches.items()}


@dataclass(slots=True)
class CyclePolicy(Policy):
    """Round-robin branch selection (local, deterministic, memoryful)."""

    name: str = "cycle"
    _idx: dict[str, int] = field(default_factory=dict)

    def choose(self, ctx: PolicyContext) -> PolicyAction:
        out: PolicyAction = {}
        for i, branches in ctx.intersection_to_branches.items():
            branches = sorted(branches)
            if not branches:
                out[i] = None
                continue
            j = self._idx.get(i, 0) % len(branches)
            out[i] = branches[j]
            self._idx[i] = j + 1
        return out


class LongestCurrentWaitPolicy(Policy):
    """Chooses incoming branch with largest current queue age max (local, deterministic tie-break)."""

    name = "longest_current_wait"

    def choose(self, ctx: PolicyContext) -> PolicyAction:
        out: PolicyAction = {}
        for i, branches in ctx.intersection_to_branches.items():
            if not branches:
                out[i] = None
                continue
            out[i] = max(sorted(branches), key=lambda b: (ctx.queue_waits.get((i, b), 0), -ord(b[0])))
        return out


class LongestCumulativeWaitPolicy(Policy):
    """Chooses incoming branch with largest cumulative wait (local, deterministic tie-break)."""

    name = "longest_cumulative_wait"

    def choose(self, ctx: PolicyContext) -> PolicyAction:
        out: PolicyAction = {}
        for i, branches in ctx.intersection_to_branches.items():
            if not branches:
                out[i] = None
                continue
            out[i] = max(sorted(branches), key=lambda b: (ctx.queue_cumulative_waits.get((i, b), 0), -ord(b[0])))
        return out


POLICIES = {
    "all_open": AllOpenPolicy,
    "cycle": CyclePolicy,
    "longest_current_wait": LongestCurrentWaitPolicy,
    "longest_cumulative_wait": LongestCumulativeWaitPolicy,
}


def build_policy(name: str) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"Unknown policy: {name}")
    return POLICIES[name]()
