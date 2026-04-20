# evacuation-vug

A clean Python package for directed-tree evacuation queue simulation plus the revised valid-utility-game (VUG) surrogate used in the updated paper framing.

## What this package does

- Simulates queueing on a directed tree with generator nodes, intersection controllers, and one safe sink.
- Runs four baseline policies:
  - `all_open`
  - `cycle`
  - `longest_current_wait`
  - `longest_cumulative_wait`
- Computes operational outcomes (completion rate, travel times, waits).
- Computes VUG surrogate route counts, welfare, and intersection marginal utilities.
- Runs exact tiny-game equilibrium enumeration and approximate best-response dynamics.
- Produces paper-style figures and CSV/JSON summaries in `results/`.

## Queueing layer vs surrogate layer

The **queueing simulator** tracks realized movement and delay.

The **surrogate game layer** evaluates service-token schedules:

- token: `r = (i, b, t)`
- route count: `N_k(S) = |{(i,b_i(k),t) in S}|`
- welfare: `W(S) = sum_k alpha_k phi_k(N_k(S))`
- utility: `u_i(S) = W(S) - W(S_{-i})`

Supported `phi_k` families:

1. capped: `phi_k(n)=min(n,D_k)`
2. geometric: `phi_k(n)=sum_{m=1}^n beta_k^(m-1)` for `0<beta_k<=1`

## Install

```bash
python -m pip install -e .
python -m pip install -e .[dev]
```

## CLI quick start

```bash
python -m evacuation_vug.cli run-experiment --config configs/experiments/subcritical.yaml
python -m evacuation_vug.cli reproduce-paper
python -m evacuation_vug.cli compute-small-game-poa --config configs/experiments/toy.yaml
```

Outputs go to per-regime directories under `results/`.

## Reproducibility

- Single seed in each experiment config.
- Scripts log deterministic outputs to CSV/JSON + PNG/PDF figures.
- `scripts/reproduce_paper_figures.py` regenerates all regime outputs.

## Exact vs approximate

- **Exact**: `equilibria.enumerate_pure_nash` (tiny instances only).
- **Approximate**: `equilibria.best_response_dynamics` (larger instances heuristic).

## Notes on ambiguity

The exact original paper topology and numeric replication are not claimed here. The default tree and parameters are configurable and designed to preserve the original flavor while keeping the code reusable and testable.
