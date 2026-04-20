from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import log

from evacuation_vug.vehicles import Vehicle


@dataclass(slots=True)
class EdgeQueue:
    edge: tuple[str, str]
    capacity: int
    hold_base: float
    hold_log_coeff: float
    vehicles: deque[Vehicle] = field(default_factory=deque)

    def occupancy(self) -> int:
        return len(self.vehicles)

    def enqueue(self, vehicle: Vehicle) -> bool:
        if self.occupancy() >= self.capacity:
            return False
        vehicle.queue_age = 0
        self.vehicles.append(vehicle)
        return True

    def min_hold_time(self) -> int:
        s = max(self.occupancy(), 1)
        return int(round(self.hold_base + self.hold_log_coeff * log(s)))

    def front_eligible(self) -> bool:
        if not self.vehicles:
            return False
        return self.vehicles[0].queue_age >= self.min_hold_time()

    def pop_eligible(self) -> Vehicle | None:
        if self.front_eligible():
            return self.vehicles.popleft()
        return None

    def current_wait(self) -> int:
        return max((v.queue_age for v in self.vehicles), default=0)

    def cumulative_wait(self) -> int:
        return sum(v.queue_age for v in self.vehicles)
