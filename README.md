# Evacuation Traffic Queuing Simulation

A simulation framework for studying emergency evacuation traffic on arterial road networks using queueing models and traffic-signal control. The project compares local and coordinated intersection policies under extreme demand to understand congestion, fairness, and systemic inefficiencies during evacuations.
This repository contains the simulation code accompanying the paper  
**“Traffic Queuing and Signal Control in Emergency Evacuations”** (BYU CS 501R final project).  
The project studies evacuation traffic on hierarchical road networks using queueing models and traffic signal control, with an emphasis on fairness, congestion, and coordination under extreme demand.

## Motivation

Large-scale emergency evacuations often overload arterial road networks, producing gridlock even when total road capacity is theoretically sufficient. During events such as the 2017 Oroville Dam evacuation, traffic signals operating under standard local control policies contributed to severe delays and inequitable outcomes across routes.

This project explores how different traffic signal control policies perform during evacuations, and whether decentralized intersection-level decision-making can be expected to approximate socially optimal behavior.

## Modeling Framework

- **Network structure:**  
  Directed tree networks representing hierarchical evacuation routes converging to a common safe node.

- **Vehicles:**  
  Generated stochastically at upstream nodes and traverse the network toward safety.

- **Intersections:**  
  Modeled as signal-controlled merging points that regulate access to downstream queues.

- **Dynamics:**  
  Queue evolution is simulated discretely, capturing congestion, merging conflicts, and bottleneck formation.

## Signal Control Policies

The simulation compares multiple traffic signal strategies, including:
- Local queue-based policies
- Fixed or heuristic coordination policies
- Policies prioritizing fairness (e.g., minimizing worst-case delay)
- Policies prioritizing throughput or average travel time

No policy is assumed to be optimal a priori; performance depends strongly on traffic generation rate and network structure.

## Evaluation Metrics

The code evaluates both **individual-level** and **system-level** costs, including:
- Total evacuation time
- Maximum and average queue waiting times
- Worst-case vehicle delay (fairness)
- Distributional effects under congestion
- Emergent inefficiencies from decentralized control

The analysis investigates whether evacuation signal control can be modeled as a *minimization valid utility game* and examines the existence (or failure) of price-of-anarchy guarantees.

## Key Findings (from the paper)

- Local signal policies can induce highly uneven delays across routes under heavy demand.
- Coordinated policies may reduce worst-case outcomes but often increase average travel times.
- Standard game-theoretic efficiency guarantees do not directly apply to evacuation traffic.
- Trade-offs between fairness and throughput are unavoidable near critical load regimes.

## Repository Structure

```text
evacuation-traffic-games/
├─ notebooks/
│  └─ evac_simulation.ipynb          # original Colab/interactive exploration
├─ scripts/
│  └─ run_experiments.py             # command-line entrypoint to reproduce figures/results
├─ src/
│  └─ evacsim/
│     ├─ __init__.py                 # public API re-exports (see below)
│     ├─ sim.py                      # network generation + simulation loop(s)
│     ├─ policies.py                 # signal-control policies
│     └─ plotting.py                 # plotting utilities for paper-style figures
├─ CS_501R_Final_Project_Traffic_Queuing__IEEE_.pdf  # paper/report
├─ pyproject.toml                    # packaging config (editable install support)
├─ README.md
└─ .gitignore
```
## Usage

To use this package locally, install it directly from this github repository:

`pip install "git+https://github.com/perryGabriel/evacuation-traffic-games.git"`

The script `scripts/run_expiriments.py` provided in this repository is a minimal working example to replicate the results from the paper, included as a PDF in this repository.

## Citation

The Bibtex citation for this repository is:

```
@misc{perryGabriel_evacuation_traffic_games,
  author       = {Gabriel Perry},
  title        = {Evacuation Traffic Games},
  year         = {2025},
  howpublished = {\url{https://github.com/perryGabriel/evacuation-traffic-games}},
  note         = {Accessed: 2025-12-17}
}
```
