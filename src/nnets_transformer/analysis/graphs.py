"""Versioned influence graphs (influence-v2).

Rows intervene, columns observe. Strength is 1 - cosine similarity, transformed
once; shortest paths use a separate positive distance. The inventory and
architecture are inputs, never inferred from retained edges.
"""
from dataclasses import dataclass
import math
import re

import networkx as nx

CORRECTED = "influence-v2"


@dataclass(frozen=True)
class Component:
    id: str
    layer: int
    site: str
    head: int | None = None

    @classmethod
    def parse(cls, name: str, n_layers: int):
        if name in {"hook_embed", "hook_pos_embed"}:
            return cls(name, -1, name)
        if name == "logit":
            return cls(name, n_layers + 1, name)
        if name.startswith("ln_final."):
            return cls(name, n_layers, name)
        match = re.fullmatch(r"blocks\.(\d+)\.(attn|mlp)\.hook_(\w+?)(?:\.(\d+))?", name)
        if not match:
            raise ValueError(f"Unsupported component: {name}")
        layer, module, site, head = match.groups()
        if int(layer) >= n_layers:
            raise ValueError(f"Layer outside inventory: {name}")
        return cls(name, int(layer), f"{module}.{site}", int(head) if head is not None else None)


def admissible(source: Component, target: Component, architecture: str):
    """Reachability for existing hook sites, respecting parallel attention/MLP.

    Head identity constrains q/k/v -> rotary/z paths within a layer. Whole
    components overlap head nodes, so callers choose one resolution.
    """
    if architecture not in {"parallel", "sequential"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    if source.layer != target.layer:
        return source.layer < target.layer
    if source.head is not None and target.head is not None and source.head != target.head:
        return False
    descendants = {
        "attn.q": {"attn.rot_q", "attn.z"},
        "attn.k": {"attn.rot_k", "attn.z"},
        "attn.v": {"attn.z"},
        "attn.rot_q": {"attn.z"},
        "attn.rot_k": {"attn.z"},
    }
    if architecture == "sequential" and source.site.startswith("attn.") and target.site == "mlp.post":
        return True
    return target.site in descendants.get(source.site, set())


def build_graph(similarities, *, inventory, threshold, version, architecture="parallel", resolution="component", epsilon=1e-8):
    """Build a directed graph from a pandas matrix without changing its values.

    All selected inventory nodes are kept (isolates included); strength and
    distance are separate. Missing measurements remain absent edges, not
    fabricated zero observations.
    """
    if version != CORRECTED:
        raise ValueError(f"Unknown graph version: {version}")
    if resolution not in {"component", "head", "mixed"}:
        raise ValueError("resolution must be component, head or mixed")
    if resolution == "mixed":
        raise ValueError("v2 requires a non-overlapping component or head resolution")
    if architecture not in {"parallel", "sequential"}:
        raise ValueError("architecture must be parallel or sequential")
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("similarity threshold must be in [-1, 1]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    components = {c.id: c for c in inventory}
    if len(components) != len(inventory):
        raise ValueError("Duplicate component IDs")
    if not similarities.index.is_unique or not similarities.columns.is_unique:
        raise ValueError("Duplicate matrix labels")
    missing = (set(similarities.index) | set(similarities.columns)) - components.keys()
    if missing:
        raise ValueError(f"Matrix labels missing from inventory: {sorted(missing)}")
    head_sites = {c.id.rsplit(".", 1)[0] for c in inventory if c.head is not None}
    selected = {k: c for k, c in components.items() if resolution == "mixed" or
                (resolution == "component" and c.head is None) or
                (resolution == "head" and (c.head is not None or c.id not in head_sites))}
    graph = nx.DiGraph(graph_version=version, resolution=resolution, architecture=architecture,
                       threshold=threshold, strength="1-cosine", distance="inverse_strength",
                       epsilon=epsilon, inventory="complete_selected_input_inventory",
                       ordering="architecture-reachability-v2",
                       threshold_rule="similarity < threshold", missing_measurements="excluded",
                       metric_version="inverse-strength-paths-v2")
    graph.add_nodes_from((c.id, {"layer": c.layer, "head": c.head, "site": c.site}) for c in selected.values())
    possible = 0
    measured = 0
    for source in selected.values():
        for target in selected.values():
            allowed = admissible(source, target, architecture)
            possible += int(allowed)
            if not allowed or source.id not in similarities.index or target.id not in similarities.columns:
                continue
            value = float(similarities.at[source.id, target.id])
            if math.isnan(value):
                continue
            if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
                raise ValueError(f"Invalid cosine at {source.id} -> {target.id}: {value}")
            measured += 1
            if value >= threshold:
                continue
            strength = 1 - value
            graph.add_edge(source.id, target.id, similarity=value, weight=strength, strength=strength,
                           distance=1 / (strength + epsilon))
    graph.graph["admissible_pairs"] = possible
    graph.graph["measured_admissible_pairs"] = measured
    graph.graph["measurement_coverage"] = measured / possible if possible else None
    graph.graph["measurement_status"] = "complete" if measured == possible else "partial"
    graph.graph["active_nodes"] = sum(d > 0 for _, d in graph.degree())
    graph.graph["admissible_density"] = graph.number_of_edges() / possible if possible else None
    return graph


def centralities(graph):
    """Incoming/outgoing directed closeness over positive path distances."""
    distance = "distance"
    return {
        "in_strength": dict(graph.in_degree(weight="weight")),
        "out_strength": dict(graph.out_degree(weight="weight")),
        "betweenness": nx.betweenness_centrality(graph, weight=distance),
        "closeness_in": nx.closeness_centrality(graph, distance=distance),
        "closeness_out": nx.closeness_centrality(graph.reverse(copy=False), distance=distance),
        "pagerank": nx.pagerank(graph, weight="weight"),
    }
