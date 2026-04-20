from __future__ import annotations

import argparse
from pathlib import Path

from evacuation_vug.config import load_experiment_config
from evacuation_vug.equilibria import best_response_dynamics, enumerate_pure_nash
from evacuation_vug.io import ensure_dir, write_json, write_rows_csv
from evacuation_vug.metrics import policy_summary
from evacuation_vug.network import load_network
from evacuation_vug.plotting import (
    plot_generator_histograms,
    plot_network_policy_panels,
    plot_poa,
    plot_utility_heatmap,
    plot_welfare_bar,
)
from evacuation_vug.policies import build_policy
from evacuation_vug.simulator import Simulator


def run_experiment(config_path: str) -> dict[str, dict]:
    cfg = load_experiment_config(config_path)
    network = load_network(cfg.network_config)
    outdir = ensure_dir(cfg.output_dir)
    summaries: dict[str, dict] = {}
    for policy_name in cfg.policies:
        sim = Simulator(network=network, generation_rates=cfg.generation, seed=cfg.seed)
        result = sim.run(cfg.horizon, build_policy(policy_name))
        summary = policy_summary(network, result, cfg.surrogate)
        summaries[policy_name] = summary

    rows = [{"policy": p, **{k: v for k, v in s.items() if isinstance(v, (int, float, str))}} for p, s in summaries.items()]
    write_rows_csv(outdir / "policy_summary.csv", rows)
    write_json(outdir / "policy_summary.json", summaries)
    plot_network_policy_panels(network, summaries, outdir / "network_summary")
    plot_generator_histograms(network, summaries, outdir / "generator_histograms")
    plot_welfare_bar(summaries, outdir / "welfare_by_policy")
    plot_utility_heatmap(summaries, outdir / "utility_heatmap")
    return summaries


def reproduce_paper() -> None:
    for cfg in [
        "configs/experiments/subcritical.yaml",
        "configs/experiments/critical.yaml",
        "configs/experiments/supercritical.yaml",
    ]:
        run_experiment(cfg)


def compute_small_game_poa(config_path: str) -> dict:
    cfg = load_experiment_config(config_path)
    network = load_network(cfg.network_config)
    exact = enumerate_pure_nash(network, horizon=min(3, cfg.horizon), surrogate_cfg=cfg.surrogate)
    approx = best_response_dynamics(network, horizon=min(5, cfg.horizon), surrogate_cfg=cfg.surrogate, seed=cfg.seed)
    outdir = ensure_dir(cfg.output_dir)
    payload = {
        "equilibria_count": len(exact.equilibria),
        "optimal_welfare": exact.optimal_welfare,
        "best_equilibrium_welfare": exact.best_equilibrium_welfare,
        "poa": exact.poa,
        "approximate": approx,
    }
    write_json(outdir / "small_game_poa.json", payload)
    plot_poa([payload], outdir / "poa_plot")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evacuation VUG experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run-experiment")
    p1.add_argument("--config", required=True)

    p2 = sub.add_parser("reproduce-paper")

    p3 = sub.add_parser("compute-small-game-poa")
    p3.add_argument("--config", required=True)

    args = parser.parse_args()

    if args.cmd == "run-experiment":
        run_experiment(args.config)
    elif args.cmd == "reproduce-paper":
        reproduce_paper()
    elif args.cmd == "compute-small-game-poa":
        compute_small_game_poa(args.config)


if __name__ == "__main__":
    main()
