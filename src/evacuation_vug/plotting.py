from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from evacuation_vug.network import TreeNetwork


def _save(fig: plt.Figure, path_base: str | Path, dpi: int = 250) -> None:
    p = Path(path_base)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_network_policy_panels(network: TreeNetwork, summaries: dict[str, dict], out_base: str | Path) -> None:
    policies = list(summaries)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pos = nx.spring_layout(network.graph, seed=7)
    for ax, p in zip(axes.ravel(), policies):
        nx.draw(network.graph, pos=pos, with_labels=True, node_size=600, font_size=8, ax=ax)
        s = summaries[p]
        ax.set_title(f"{p}\nW={s['surrogate_welfare']:.2f}, completion={100*s['completion_pct']:.1f}%")
    _save(fig, out_base)


def plot_generator_histograms(network: TreeNetwork, summaries: dict[str, dict], out_base: str | Path) -> None:
    policies = list(summaries)
    gens = network.generators
    fig, axes = plt.subplots(len(gens), len(policies), figsize=(4 * len(policies), 2.5 * len(gens)), squeeze=False)
    for r, g in enumerate(gens):
        for c, p in enumerate(policies):
            ax = axes[r][c]
            vals = summaries[p].get("generator_travel", {}).get(g, {})
            series = vals.get("raw", []) if isinstance(vals, dict) else []
            if series:
                ax.hist(series, bins=20, alpha=0.8)
                ax.axvline(np.mean(series), linestyle="--")
                ax.axvline(np.max(series), linestyle=":")
            ax.set_title(f"{g} | {p}")
    _save(fig, out_base)


def plot_welfare_bar(summaries: dict[str, dict], out_base: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    policies = list(summaries)
    vals = [summaries[p]["surrogate_welfare"] for p in policies]
    ax.bar(policies, vals)
    ax.set_ylabel("Surrogate welfare W")
    _save(fig, out_base)


def plot_utility_heatmap(summaries: dict[str, dict], out_base: str | Path) -> None:
    policies = list(summaries)
    intersections = sorted({i for s in summaries.values() for i in s["player_utilities"].keys()})
    arr = np.array([[summaries[p]["player_utilities"].get(i, 0.0) for i in intersections] for p in policies])
    fig, ax = plt.subplots(figsize=(1 + len(intersections), 1 + len(policies)))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(intersections)), intersections, rotation=45)
    ax.set_yticks(range(len(policies)), policies)
    fig.colorbar(im, ax=ax, label="u_i")
    _save(fig, out_base)


def plot_poa(points: list[dict], out_base: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(points)))
    y = [p["poa"] for p in points]
    ax.plot(x, y, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical PoA")
    ax.set_xlabel("Toy game index")
    _save(fig, out_base)
