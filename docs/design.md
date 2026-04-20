# Design

The repository separates three layers:

1. **Network/config layer** (`config.py`, `network.py`) for YAML-driven directed-tree definitions.
2. **Operational simulator layer** (`vehicles.py`, `queues.py`, `simulator.py`, `policies.py`) for realized delay and throughput.
3. **Surrogate game layer** (`surrogate.py`, `game.py`, `equilibria.py`) for token-based welfare/utility and equilibrium analysis.

`metrics.py`, `plotting.py`, and `cli.py` integrate outputs into reproducible experiment workflows.
