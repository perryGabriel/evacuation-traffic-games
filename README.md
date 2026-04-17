# Evacuation Traffic Queuing Simulation

A Python simulation framework for studying emergency evacuation traffic on directed road networks with queue-based dynamics and signal-control policies.

This repository contains the code used for the BYU CS 501R project paper:
**“Traffic Queuing and Signal Control in Emergency Evacuations.”**

---

## Table of Contents

- [What this project models](#what-this-project-models)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running experiments (CLI)](#running-experiments-cli)
- [Using the package API](#using-the-package-api)
- [Policies included](#policies-included)
- [Simulation outputs and metrics](#simulation-outputs-and-metrics)
- [Generated artifacts](#generated-artifacts)
- [Reproducing the notebook workflow](#reproducing-the-notebook-workflow)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## What this project models

The simulator represents an evacuation road system as a directed graph whose edges are traffic queues and whose nodes are:

- **Generators** (upstream source points where vehicles are created),
- **Intersections** (signal-controlled merge points), and
- **Safe-region sink node(s)** (destination).

At each timestep, the simulator:

1. Generates cars at source nodes (stochastically),
2. Applies a control policy at each intersection,
3. Moves eligible cars across queues subject to capacity constraints,
4. Updates waiting/travel times,
5. Logs per-car completion statistics and per-intersection costs.

The goal is to compare how local and coordinated policies trade off throughput, average delay, and fairness under high demand.

---

## Repository layout

```text
evacuation-traffic-games/
├─ notebooks/
│  └─ evac_simulation.ipynb
├─ scripts/
│  └─ run_experiments.py
├─ src/
│  └─ evacsim/
│     ├─ __init__.py
│     ├─ sim.py
│     ├─ policies.py
│     └─ plotting.py
├─ CS_501R_Final_Project_Traffic_Queuing__IEEE_.pdf
├─ pyproject.toml
└─ README.md
```

---

## Requirements

- Python **3.10+**
- Core dependencies:
  - `numpy`
  - `networkx`
  - `matplotlib`

---

## Installation

### Option 1: Install from GitHub

```bash
pip install "git+https://github.com/perryGabriel/evacuation-traffic-games.git"
```

### Option 2: Local editable install (recommended for development)

```bash
git clone https://github.com/perryGabriel/evacuation-traffic-games.git
cd evacuation-traffic-games
pip install -e .
```

---

## Quick start

Run the experiment script with required arguments:

```bash
python scripts/run_experiments.py \
  --num-timesteps 2000 \
  --generation-rate 0.167
```

This runs all bundled policies on one generated network and writes PDF figures to the current working directory.

---

## Running experiments (CLI)

The script `scripts/run_experiments.py` is the main command-line entrypoint.

### Required arguments

- `--num-timesteps` (`int`): number of timesteps to simulate.
- `--generation-rate` (`float`): per-timestep generation probability for each generator node.

### Optional arguments

- `--num-nodes` (`int`, default `15`): number of nodes in the generated network.
- `--seed` (`int`, default `42`): random seed for network generation/layout consistency.
- `--dpi` (`int`, default `300`): figure output DPI.
- `--verbose` (`int`, default `0`): verbosity level.

### Full example

```bash
python scripts/run_experiments.py \
  --num-timesteps 3000 \
  --generation-rate 0.20 \
  --num-nodes 20 \
  --seed 7 \
  --dpi 300 \
  --verbose 1
```

---

## Using the package API

You can also run simulations directly from Python.

```python
from evacsim.sim import build_road_network, run_multiple_simulations
from evacsim.policies import (
    all_open_policy,
    cycle_policy,
    longest_current_wait_policy,
    longest_cumulative_wait_policy,
)

G = build_road_network(num_nodes=15, seed=42)
generation_rates = {n: 0.167 for n in G.nodes() if G.in_degree(n) == 0}

grouped_travel_times, intersection_costs, pct_finished = run_multiple_simulations(
    G,
    num_timesteps=2000,
    all_policies_list=[
        all_open_policy,
        cycle_policy,
        longest_current_wait_policy,
        longest_cumulative_wait_policy,
    ],
    generation_rates=generation_rates,
)
```

### Core API functions

- `evacsim.sim.generate_random_DAG(num_nodes, seed=None)`
- `evacsim.sim.build_road_network(edges=None, num_nodes=None, pos=None, seed=None, verbose=0)`
- `evacsim.sim.run_simulation(G, num_timesteps, policies, generation_rates, wait_time_func=..., verbose=0)`
- `evacsim.sim.run_multiple_simulations(G, num_timesteps, all_policies_list, generation_rates, verbose=0)`
- `evacsim.plotting.plot_travel_times_by_policy(...)`
- `evacsim.plotting.plot_network_with_generator_stats(...)`

---

## Policies included

The repository currently includes these policy functions in `src/evacsim/policies.py`:

- `all_open_policy`: opens all incoming queues.
- `cycle_policy`: serves one incoming queue at a time in round-robin order.
- `longest_current_wait_policy`: serves the queue with the largest in-queue wait.
- `longest_cumulative_wait_policy`: serves the queue with the largest cumulative travel-time leader.
- `greedy_longest_current_wait_policy`: serves based on front-car wait only.

You can provide your own policy callable(s) as long as they mutate each intersection's
`current_state_incoming_queue_ids` consistently with the simulator expectations.

---

## Simulation outputs and metrics

`run_simulation(...)` returns:

1. `completed_cars_log`: list of dictionaries with `car_id`, `origin`, and `total_travel_time`.
2. `queue_fullness_history`: per-timestep queue occupancies.
3. `intersection_costs_history`: per-timestep intersection cost snapshots.
4. `percent_cars_finished`: completed / (completed + in-system) by final timestep.

`run_multiple_simulations(...)` returns grouped policy-comparison structures:

1. `grouped_travel_times` keyed by policy name and generator.
2. `exp_int_costs_by_policy` with cost trajectories per intersection.
3. `percent_cars_finished_by_policy` keyed by policy name.

---

## Generated artifacts

By default, `scripts/run_experiments.py` writes:

- `gen_hist_<rate>.pdf`: generator-by-policy travel-time histograms.
- `net_states_<rate>.pdf`: network plots with generator/intersection statistics.

Files are written in the working directory where the script is run.

---

## Reproducing the notebook workflow

An exploratory notebook is included at:

- `notebooks/evac_simulation.ipynb`

Use it for interactive experimentation, policy prototyping, and visual inspection in a Jupyter/Colab-style environment.

---

## Assumptions and limitations

- The generated road network is tree-like and directed toward sink node `0`.
- Queue capacity is fixed per edge (`TrafficQueue`, default capacity `5` unless overridden).
- Generator insertion currently uses `override_capacity=True` at source queues.
- Time is discretized; movement and waiting updates occur in fixed order each timestep.
- Cost definitions in plotting/aggregation are tailored to the project paper and may not match all transportation-engineering conventions.

If you are extending this work to new network classes (e.g., cyclic graphs, multiple sinks, adaptive capacities), review `src/evacsim/sim.py` first.

---

## Troubleshooting

- **`ModuleNotFoundError: evacsim`**
  - Install in editable mode: `pip install -e .`
- **No output figures produced**
  - Ensure you used required CLI args and have write permission in the current directory.
- **Low completion percentages**
  - Try lower generation rates and/or more timesteps; high load can intentionally induce persistent queues.

---

## Citation

```bibtex
@misc{perryGabriel_evacuation_traffic_games,
  author       = {Gabriel Perry},
  title        = {Evacuation Traffic Games},
  year         = {2025},
  howpublished = {\url{https://github.com/perryGabriel/evacuation-traffic-games}},
  note         = {Accessed: 2026-04-17}
}
```
