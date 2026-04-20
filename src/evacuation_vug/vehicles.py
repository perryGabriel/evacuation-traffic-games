from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Vehicle:
    vehicle_id: int
    origin: str
    birth_t: int
    route: list[str]
    route_index: int = 0
    total_age: int = 0
    queue_age: int = 0
    completion_time: int | None = None

    def tick(self) -> None:
        self.total_age += 1
        self.queue_age += 1

    @property
    def completed(self) -> bool:
        return self.completion_time is not None
