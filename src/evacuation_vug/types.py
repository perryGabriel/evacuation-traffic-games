from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Token = tuple[str, str, int]
PolicyAction = dict[str, str | None]


@dataclass(slots=True)
class GeneratorSpec:
    node: str
    arrival_rate: float


@dataclass(slots=True)
class QueueSpec:
    capacity: int = 5
    hold_base: float = 2.0
    hold_log_coeff: float = 2.0


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    horizon: int
    seed: int
    network_config: str
    generation: dict[str, float]
    surrogate: dict
    policies: list[str]
    output_dir: str


PhiFunction = Callable[[int], float]


@dataclass(slots=True)
class SurrogateSpec:
    alpha: dict[str, float]
    phi_type: dict[str, str]
    phi_params: dict[str, dict[str, float]] = field(default_factory=dict)
