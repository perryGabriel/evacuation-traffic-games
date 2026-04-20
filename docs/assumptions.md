# Assumptions

- Directed topology has underlying undirected tree structure.
- One distinguished safe node.
- Discrete-time dynamics and finite-horizon game abstraction.
- Intersection controls at most one incoming branch per step.
- Edge queue service eligibility uses occupancy-dependent minimum hold time:
  - approximately `2 + 2 ln(S)` by default.
- Service tokens are extracted from chosen branch schedules.
- Exact PoA is tractable only for tiny horizons/intersection counts due to combinatorics.
- Larger settings use heuristic best-response dynamics and are reported as approximate.
