from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from evacuation_vug.config import load_yaml


@dataclass(slots=True)
class TreeNetwork:
    graph: nx.DiGraph
    safe_node: str
    generators: list[str]
    intersections: list[str]

    def route_to_safe(self, generator: str) -> list[str]:
        return nx.shortest_path(self.graph, generator, self.safe_node)

    def incoming_branches(self, intersection: str) -> list[str]:
        return sorted(str(u) for u, _ in self.graph.in_edges(intersection))

    def branch_id(self, intersection: str, upstream: str) -> str:
        return f"{upstream}->{intersection}"

    def route_branch_for_generator(self, generator: str) -> dict[str, str]:
        route = self.route_to_safe(generator)
        mapping: dict[str, str] = {}
        for u, v in zip(route[:-1], route[1:]):
            if v in self.intersections:
                mapping[v] = self.branch_id(v, u)
        return mapping


def _validate_tree(net: TreeNetwork) -> None:
    g = net.graph
    if net.safe_node not in g:
        raise ValueError("safe node missing")
    und = g.to_undirected()
    if not nx.is_tree(und):
        raise ValueError("graph must be an underlying tree")
    for gen in net.generators:
        if not nx.has_path(g, gen, net.safe_node):
            raise ValueError(f"generator {gen} has no path to safe node")
        if len(list(nx.all_simple_paths(g, gen, net.safe_node))) != 1:
            raise ValueError(f"generator {gen} does not have a unique path")


def load_network(path: str | Path) -> TreeNetwork:
    raw = load_yaml(path)
    g = nx.DiGraph()
    for node, attrs in raw["nodes"].items():
        g.add_node(node, **attrs)
    for e in raw["edges"]:
        g.add_edge(
            e["u"],
            e["v"],
            capacity=int(e.get("capacity", 5)),
            hold_base=float(e.get("hold_base", 2.0)),
            hold_log_coeff=float(e.get("hold_log_coeff", 2.0)),
        )
    safe = raw["safe_node"]
    generators = [n for n, d in g.nodes(data=True) if d.get("kind") == "generator"]
    intersections = [n for n, d in g.nodes(data=True) if d.get("kind") == "intersection"]
    network = TreeNetwork(g, safe, sorted(generators), sorted(intersections))
    _validate_tree(network)
    return network
