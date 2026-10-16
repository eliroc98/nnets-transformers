#!/usr/bin/env python3
"""
Figure generation for the complex-network section.

Run:
    python cn_figures.py --data . --out figures
    python cn_figures.py --data . --out figures --only turnover regimes

Inputs (produced by the analysis notebook):
    all_metrics.pickle        per (step, threshold): |V|, |E|, density, logit
    all_distributions.pickle  per (step, threshold, component): the four metrics
    edge_weights.pickle       per (step, threshold): one row per edge weight

Outputs, in two groups.

  Continuous-quantity figures (unchanged in kind from the originals, but on a
  log step axis, using every checkpoint, and without hidden outliers):
    topology_evolution.pdf
    degree_and_weight_distributions.pdf
    centrality_distributions.pdf

  Replacements for the binary >95th-percentile heatmaps. Each answers one of
  the two research questions directly rather than leaving it to the eye:
    role_turnover.pdf     RQ2: how much does upper-tail membership change per
                          step, and does it converge? Peaks in the churn curve
                          are reallocation events; the asserted phase
                          boundaries are drawn on top so the claim that
                          reallocation concentrates at them can be checked
                          rather than assumed.
    regime_similarity.pdf RQ1: checkpoint x checkpoint set-overlap matrix.
                          Phases appear as diagonal blocks, derived from the
                          data instead of drawn on by hand. This is the figure
                          that makes the three-phase claim without asserting
                          it.
    role_streams.pdf      Who holds each role, with identity kept: one lane per
                          component, height proportional to the share of the
                          upper tail it holds. --streams-style stack gives the
                          stacked-area variant instead.
    role_profile.pdf      Components x (metric, phase): which component holds
                          which role, and in which phase.
    layer_occupancy.pdf   Where in the depth of the model the upper tail sits,
                          as a fraction contributed by each layer. Eight rows
                          instead of fifty, and it states the depth-migration
                          claim directly.
    component_traces.pdf  Magnitude, not membership: the actual metric value
                          of the few components that hold a role longest,
                          against the rest in grey.
    heatmap_<metric>.pdf  The original view, kept for the appendix, but
                          seriated (rows ordered by the temporal centroid of
                          occupancy) so that succession reads as a staircase,
                          with layer rules and bold module-level labels.
"""

from __future__ import annotations

import argparse
import colorsys
import os
import pickle
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

THRESHOLDS = [0.1, 0.3, 0.5, 0.7]
TAU_MAIN = 0.7
PERCENTILE = 0.95

# Phase boundaries used for annotation. The first separates the log-spaced
# initialisation checkpoints from the regular grid; the other two are the
# boundaries read off the topology. They are drawn, never used to compute
# anything, so that figures which recover them do so independently.
PHASE_EDGES = [1000, 10000, 80000]
PHASE_SPANS = [(1, 1000), (1000, 10000), (10000, 80000), (80000, 200000)]
PHASE_NAMES = ["initialisation", "exploration", "consolidation", "refinement"]
# One hue per phase, held at a common lightness so no band looks heavier than
# its neighbours: the phase is read off the colour and named once in the key,
# so nothing has to be written over the panels. The hues are spread right
# around the wheel (neutral, blue, rose, green) and saturated as far as the
# data lines on top will allow. Rose rather than the obvious apricot for
# consolidation: apricot sits under the orange out-degree trace in the
# turnover figure and swallows it.
PHASE_SHADE = ["#e0dedb", "#c7dcf2", "#f7d3d9", "#c9e7cf"]

METRICS = [
    ("in_strength_dist", "weighted in-degree"),
    ("out_strength_dist", "weighted out-degree"),
    ("betweenness_dist", "betweenness centrality"),
    ("closeness_dist", "closeness centrality"),
]

RC = {
    "pdf.fonttype": 42,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

COLOURS = {t: c for t, c in zip(THRESHOLDS, plt.cm.viridis([0.04, 0.36, 0.62, 0.86]))}
STREAMS_STYLE = "lanes"

METRIC_COLOURS = dict(zip([m for m, _ in METRICS],
                          ["#1b4f72", "#b9770e", "#7d3c98", "#1a6b3c"]))


# --------------------------------------------------------------------------
# loading and shared helpers
# --------------------------------------------------------------------------

def load(data_dir: str):
    paths = {
        "metrics": os.path.join(data_dir, "all_metrics.pickle"),
        "distributions": os.path.join(data_dir, "all_distributions.pickle"),
        "edges": os.path.join(data_dir, "edge_weights.pickle"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "missing input pickles: " + ", ".join(missing) +
            "\nPass the directory containing them with --data."
        )
    with open(paths["metrics"], "rb") as fh:
        dm = pd.DataFrame(pickle.load(fh))
    with open(paths["distributions"], "rb") as fh:
        dd = pd.DataFrame(pickle.load(fh))
    with open(paths["edges"], "rb") as fh:
        de = pd.DataFrame(pickle.load(fh))
    for name, frame in [("metrics", dm), ("distributions", dd), ("edges", de)]:
        if frame.empty:
            raise ValueError(f"{name} pickle contained no rows")
    return dm, dd, de


def smooth(series: pd.Series, w: int = 3) -> pd.Series:
    return series.rolling(w, center=True, min_periods=1).median()


def shade_phases(ax):
    for (lo, hi), col in zip(PHASE_SPANS, PHASE_SHADE):
        ax.axvspan(lo, hi, color=col, zorder=0, lw=0)
    for edge in PHASE_EDGES:
        ax.axvline(edge, color="0.55", lw=0.6, ls=(0, (4, 2)), zorder=1)


def phase_handles():
    return [Patch(facecolor=c, edgecolor="0.80", lw=0.4, label=n)
            for n, c in zip(PHASE_NAMES, PHASE_SHADE)]


def bottom_legends(fig, series=None, y: float = -0.01, ncol: int = 4,
                   fontsize: float | None = None):
    """Series key on one row, phase-colour key on the row beneath it.

    Both are figure legends, so the tight bounding box grows to include them.
    Naming the phases here rather than over the panels keeps the last two
    apart: on a log axis refinement is a third of a decade against
    exploration's full decade, too narrow to write in.
    """
    size = fontsize if fontsize is not None else mpl.rcParams["legend.fontsize"]
    kw = dict(loc="lower center", frameon=False, columnspacing=1.6)
    if series:
        fig.legend(handles=series, ncol=ncol, bbox_to_anchor=(0.5, y),
                   handlelength=1.4, fontsize=size, **kw)
        y -= (size + 5) / 72 / fig.get_figheight()   # one line, in fig fraction
    fig.legend(handles=phase_handles(), ncol=len(PHASE_NAMES),
               bbox_to_anchor=(0.5, y), handlelength=1.1, handleheight=1.0,
               fontsize=size - 0.5, **kw)


def style_x(ax, bottom: bool, xmax: float):
    ax.set_xscale("log")
    ax.set_xlim(1, xmax * 1.05)
    ax.grid(True, which="major", ls=":", color="0.8", zorder=0)
    if bottom:
        ax.set_xlabel("Training step (log scale)")
    else:
        ax.tick_params(labelbottom=False)


# --------------------------------------------------------------------------
# component naming
# --------------------------------------------------------------------------

def parse_component(name: str):
    """Return (layer, head, is_module_level). Head is -1 for module-level."""
    if name in ("emb", "pos_emb"):
        return -1, -1, True
    if name == "logits":
        return 90, -1, True
    if name.startswith("mlp"):
        digits = re.findall(r"\d+", name)
        if not digits:
            raise ValueError(f"cannot parse layer from MLP component {name!r}")
        return int(digits[-1]), -1, True
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "attn":
        raise ValueError(f"unrecognised component name {name!r}")
    layer = int(parts[2])
    if len(parts) == 3:
        return layer, -1, True
    return layer, int(parts[3]), False


def layer_label(layer: int) -> str:
    if layer == -1:
        return "emb"
    if layer == 90:
        return "logits"
    return f"L{layer}"


def sort_key(name: str):
    layer, head, module = parse_component(name)
    return (layer, 0 if module else 1, head, name)


# Every layer id that can come out of parse_component, fixed and evenly
# spaced around the hue wheel so that a given component gets the same colour
# in every figure regardless of which other components happen to share that
# figure with it. Built directly in HSV rather than sampled from tab10: tab10
# includes a near-zero-saturation grey swatch, which would make an entire
# layer's colours collapse together regardless of kind or head.
_LAYER_ORDER = [-1, 0, 1, 2, 3, 4, 5, 90]
_LAYER_HUE = {layer: i / len(_LAYER_ORDER) for i, layer in enumerate(_LAYER_ORDER)}
_LAYER_SAT, _LAYER_VAL = 0.72, 0.88

# Two components in the same layer (e.g. mlp_0 and attn.z.0) both parse to
# head=-1 and are otherwise indistinguishable by (layer, head) alone, so the
# "kind" -- which projection, or mlp/emb/logits -- gets its own hue offset
# off the layer's base hue. Kept well inside +/-half the layer spacing
# (1/16 = 0.0625) so a component never drifts into a neighbouring layer's hue.
_KIND_HUE_OFFSET = {
    "mlp": 0.00, "attn.q": 0.017, "attn.k": -0.017, "attn.v": 0.033,
    "attn.z": -0.033, "attn.rot_q": 0.05, "attn.rot_k": -0.05,
    "emb": 0.00, "pos_emb": 0.017, "logits": 0.00,
}


def _component_kind(name: str) -> str:
    if name.startswith("mlp"):
        return "mlp"
    if name in ("emb", "pos_emb", "logits"):
        return name
    parts = name.split(".")
    return f"{parts[0]}.{parts[1]}"


def component_colour(name: str):
    """A colour that depends only on the component's own name: hue from its
    layer and kind, brightness from its head. Two components never move
    between colours just because a different figure changed which components
    are top-n, and components that share a layer but differ in kind (an MLP
    and an attention projection, say) don't collapse onto the same colour.

    Module-level and per-head value ranges are kept strictly disjoint
    ([0.88] vs. [0.35, 0.80]) so no per-head shade can coincide with its own
    module's, whatever the head count.
    """
    layer, head, is_module = parse_component(name)
    h = (_LAYER_HUE[layer] + _KIND_HUE_OFFSET.get(_component_kind(name), 0.0)) % 1.0
    if is_module:
        s, v = _LAYER_SAT, _LAYER_VAL
    else:
        # golden-ratio increments so adjacent head indices don't land at
        # adjacent, hard-to-tell-apart shades
        frac = (head * 0.6180339887498949) % 1.0
        s = 0.55 + 0.35 * frac
        v = 0.35 + 0.45 * frac
    return colorsys.hsv_to_rgb(h, s, v)


# --------------------------------------------------------------------------
# upper-tail sets
# --------------------------------------------------------------------------

def upper_tail_sets(dd: pd.DataFrame, col: str, tau: float = TAU_MAIN,
                    pct: float = PERCENTILE):
    """For each step, the set of components strictly above the pct quantile."""
    sub = dd[dd.threshold == tau]
    if sub.empty:
        raise ValueError(f"no rows at threshold {tau}")
    bound = sub.groupby("step")[col].transform(lambda x: x.quantile(pct))
    hot = sub[sub[col] > bound]
    steps = np.array(sorted(sub.step.unique()))
    sets = {s: set() for s in steps}
    for step, group in hot.groupby("step"):
        sets[step] = set(group.component)
    return steps, sets


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return np.nan          # undefined; propagated, never silently zeroed
    return len(a & b) / len(union)


# --------------------------------------------------------------------------
# figure 1 — topology
# --------------------------------------------------------------------------

def fig_topology(dm, out):
    xmax = dm.step.max()
    fig, axes = plt.subplots(4, 1, figsize=(6.0, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.13})

    logit = dm[dm.threshold == TAU_MAIN].sort_values("step")
    ax = axes[0]
    shade_phases(ax)
    ax.plot(logit.step, logit.induction_log, color="0.25", lw=0.7, alpha=0.45)
    ax.plot(logit.step, smooth(logit.induction_log), color="0.15", lw=1.3)
    ax.set_ylabel("correct-token\nlogit")
    style_x(ax, False, xmax)

    for ax, col, lab in zip(axes[1:], ["num_nodes", "num_edges", "density"],
                            ["$|V|$", "$|E|$", "density $D$"]):
        shade_phases(ax)
        for t in THRESHOLDS:
            s = dm[dm.threshold == t].sort_values("step")
            ax.plot(s.step, s[col], color=COLOURS[t], lw=0.6, alpha=0.30)
            ax.plot(s.step, smooth(s[col]), color=COLOURS[t], lw=1.2)
        ax.set_ylabel(lab)
        style_x(ax, col == "density", xmax)

    handles = [Line2D([], [], color=COLOURS[t], lw=1.2, label=f"$\\tau={t}$")
               for t in THRESHOLDS]
    bottom_legends(fig, handles, y=-0.008)
    fig.savefig(os.path.join(out, "topology_evolution.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# figures 2 and 3 — distributions
# --------------------------------------------------------------------------

def quantile_frame(df, col, threshold):
    sub = df[df.threshold == threshold]
    g = sub.groupby("step")[col]
    return pd.DataFrame({
        "p50": g.median(),
        "p95": g.quantile(0.95),
        "p99": g.quantile(0.99),
        "skew": g.apply(lambda x: stats.skew(x)),
    }).sort_index()


def fig_distributions(dd, de, out):
    specs = [(dd, "in_strength_dist", "weighted in-degree"),
             (dd, "out_strength_dist", "weighted out-degree"),
             (de, "edge_weight_dist", "edge weight $w_{ij}$")]
    xmax = dd.step.max()
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.6), sharex=True,
                             gridspec_kw={"hspace": 0.16, "wspace": 0.28})
    for j, (src, col, title) in enumerate(specs):
        ax = axes[0, j]
        shade_phases(ax)
        for t in THRESHOLDS:
            q = quantile_frame(src, col, t)
            #ax.plot(q.index, smooth(q.p50), color=COLOURS[t], lw=1.1)
            ax.plot(q.index, smooth(q.p95), color=COLOURS[t], lw=0.9,)
                    #ls=(0, (3, 1.5)))
        ax.set_title(title, pad=4)
        if j == 0:
            ax.set_ylabel("95th percentile")
        style_x(ax, False, xmax)

        ax = axes[1, j]
        shade_phases(ax)
        for t in THRESHOLDS:
            q = quantile_frame(src, col, t)
            ax.plot(q.index, smooth(q["skew"]), color=COLOURS[t], lw=1.1)
        ax.axhline(0, color="0.5", lw=0.5)
        if j == 0:
            ax.set_ylabel("skewness")
        style_x(ax, True, xmax)

    handles = [Line2D([], [], color=COLOURS[t], lw=1.2, label=f"$\\tau={t}$")
               for t in THRESHOLDS]
    bottom_legends(fig, handles, y=-0.06)
    fig.savefig(os.path.join(out, "degree_and_weight_distributions.pdf"))
    plt.close(fig)


def fig_centrality_distributions(dd, out):
    specs = [("betweenness_dist", "betweenness $C_B$"),
             ("closeness_dist", "closeness $C_C$")]
    xmax = dd.step.max()
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    for i, (ax, (col, title)) in enumerate(zip(axes, specs)):
        shade_phases(ax)
        for t in THRESHOLDS:
            q = quantile_frame(dd, col, t)
            ax.plot(q.index, smooth(q.p95), color=COLOURS[t], lw=1.1,
                    label=f"$\\tau={t}$")
        ax.set_title(title, pad=4)
        ax.set_ylabel("95th percentile")
        style_x(ax, i == len(specs) - 1, xmax)
    axes[0].legend(ncol=2, loc="upper left", frameon=False, handlelength=1.2,
                   columnspacing=1.0, fontsize=6)
    bottom_legends(fig, y=-0.01)
    fig.savefig(os.path.join(out, "centrality_distributions.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 4 — role turnover (replaces "the identity of the tail changes")
# --------------------------------------------------------------------------

def fig_role_turnover(dd, out):
    xmax = dd.step.max()
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 4.6), sharex=True,
                             gridspec_kw={"hspace": 0.14})
    for col, title in METRICS:
        steps, sets = upper_tail_sets(dd, col)
        sizes = np.array([len(sets[s]) for s in steps], dtype=float)
        churn = np.array([np.nan] + [1 - jaccard(sets[a], sets[b])
                                     for a, b in zip(steps[:-1], steps[1:])])
        final = sets[steps[-1]]
        conv = np.array([jaccard(sets[s], final) for s in steps])
        c = METRIC_COLOURS[col]
        axes[0].plot(steps, sizes, color=c, lw=1.0, label=title)
        axes[1].plot(steps, pd.Series(churn).rolling(5, center=True,
                                                     min_periods=1).mean(),
                     color=c, lw=1.1)
        axes[2].plot(steps, pd.Series(conv).rolling(5, center=True,
                                                    min_periods=1).mean(),
                     color=c, lw=1.1)

    for ax, ylab in [(axes[0], "size of\nupper tail"),
                     (axes[1], "churn\n$1-J(T_t,T_{t-1})$"),
                     (axes[2], "convergence\n$J(T_t,T_{\\mathrm{end}})$")]:
        shade_phases(ax)
        ax.set_ylabel(ylab)
        style_x(ax, ax is axes[2], xmax)
        ax.set_zorder(2)
        ax.patch.set_visible(False)

    handles, _ = axes[0].get_legend_handles_labels()
    bottom_legends(fig, handles, y=-0.04, fontsize=6.5)
    fig.savefig(os.path.join(out, "role_turnover.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 5 — regime similarity matrices (derives the phases)
# --------------------------------------------------------------------------

def fig_regime_similarity(dd, out):
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4),
                             gridspec_kw={"hspace": 0.28, "wspace": 0.22})
    tick_steps = [1, 1000, 10000, 40000, 80000, 143000]
    last_im = None
    for ax, (col, title) in zip(axes.ravel(), METRICS):
        steps, sets = upper_tail_sets(dd, col)
        n = len(steps)
        mat = np.full((n, n), np.nan)
        for i in range(n):
            si = sets[steps[i]]
            for j in range(i, n):
                v = jaccard(si, sets[steps[j]])
                mat[i, j] = mat[j, i] = v
        last_im = ax.imshow(mat, cmap="magma", vmin=0, vmax=1,
                            origin="lower", interpolation="nearest")
        for edge in PHASE_EDGES:
            k = np.searchsorted(steps, edge) - 0.5
            ax.axvline(k, color="#2ecc71", lw=0.7)
            ax.axhline(k, color="#2ecc71", lw=0.7)
        pos = [np.searchsorted(steps, t) for t in tick_steps if t <= steps[-1]]
        labs = [f"{t:,}" for t in tick_steps if t <= steps[-1]]
        ax.set_xticks(pos)
        ax.set_xticklabels(labs, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(pos)
        ax.set_yticklabels(labs, fontsize=6)
        ax.set_title(title, fontsize=8, pad=4)
        ax.tick_params(length=2)
    fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02,
                 label="set overlap $J$ between checkpoints")
    fig.savefig(os.path.join(out, "regime_similarity.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 6 — layer occupancy (replaces the 50-row heatmaps)
# --------------------------------------------------------------------------

def fig_layer_occupancy(dd, out):
    layers = [-1, 0, 1, 2, 3, 4, 5, 90]
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(6.4, 5.4),
                             gridspec_kw={"hspace": 0.30})
    tick_steps = [1, 1000, 10000, 40000, 80000, 143000]
    last_im = None
    for ax, (col, title) in zip(axes, METRICS):
        steps, sets = upper_tail_sets(dd, col)
        mat = np.zeros((len(layers), len(steps)))
        for j, s in enumerate(steps):
            members = sets[s]
            if not members:
                mat[:, j] = np.nan
                continue
            for comp in members:
                layer, _, _ = parse_component(comp)
                mat[layers.index(layer), j] += 1
            mat[:, j] /= len(members)
        cmap = plt.get_cmap("magma_r").copy()
        cmap.set_bad("#f2f2f2")
        masked = np.ma.masked_where(~np.isfinite(mat) | (mat == 0), mat)
        last_im = ax.imshow(masked, aspect="auto", cmap=cmap,
                            vmin=0, vmax=0.6, origin="lower",
                            interpolation="nearest")
        for k in range(1, len(layers)):
            ax.axhline(k - 0.5, color="white", lw=0.5)
        for edge in PHASE_EDGES:
            ax.axvline(np.searchsorted(steps, edge) - 0.5, color="#2471a3", lw=0.8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([layer_label(l) for l in layers], fontsize=6)
        pos = [np.searchsorted(steps, t) for t in tick_steps if t <= steps[-1]]
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{t:,}" for t in tick_steps if t <= steps[-1]],
                           fontsize=6)
        ax.set_title(title, fontsize=8, pad=3)
        ax.tick_params(length=2)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Training step (checkpoints, not to scale)")
    fig.colorbar(last_im, ax=axes, fraction=0.022, pad=0.02,
                 label="share of the upper tail")
    fig.savefig(os.path.join(out, "layer_occupancy.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 7 — component traces (magnitude, not membership)
# --------------------------------------------------------------------------

def fig_component_traces(dd, out, n_named: int = 6):
    cols = ["in_strength_dist", "out_strength_dist", "closeness_dist"]
    titles = dict(METRICS)
    xmax = dd.step.max()
    fig, axes = plt.subplots(len(cols), 1, figsize=(6.0, 5.0), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    for ax, col in zip(axes, cols):
        steps, sets = upper_tail_sets(dd, col)
        occupancy = {}
        for s in steps:
            for comp in sets[s]:
                occupancy[comp] = occupancy.get(comp, 0) + 1
        if not occupancy:
            raise ValueError(f"no component ever entered the upper tail for {col}")
        named = [c for c, _ in sorted(occupancy.items(), key=lambda kv: -kv[1])][:n_named]
        sub = dd[dd.threshold == TAU_MAIN]
        wide = sub.pivot_table(index="step", columns="component", values=col)
        shade_phases(ax)
        for comp in wide.columns:
            if comp in named:
                continue
            ax.plot(wide.index, wide[comp], color="0.80", lw=0.35, alpha=0.5,
                    zorder=2)
        for comp in named:
            ax.plot(wide.index, smooth(wide[comp]), color=component_colour(comp),
                    lw=1.2, label=comp, zorder=3)
        ax.set_ylabel(titles[col].replace(" ", "\n", 1))
        finite = wide.to_numpy()
        top = np.nanmax(finite[np.isfinite(finite)])
        bottom = min(0.0, np.nanmin(finite[np.isfinite(finite)]))
        ax.set_ylim(bottom, top * 1.42 if top > 0 else 1.0)
        ax.legend(ncol=3, loc="upper left", frameon=False, fontsize=6,
                  handlelength=1.2, columnspacing=1.0)
        style_x(ax, ax is axes[-1], xmax)
        ax.set_zorder(2)
        ax.patch.set_visible(False)
    bottom_legends(fig, y=-0.01)
    fig.savefig(os.path.join(out, "component_traces.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 8 — seriated heatmaps (the original view, for the appendix)
# --------------------------------------------------------------------------

def fig_heatmap(dd, col, title, fname, out, order_by: str = "centroid"):
    steps, sets = upper_tail_sets(dd, col)
    active = sorted({c for s in steps for c in sets[s]})
    if not active:
        raise ValueError(f"no component ever entered the upper tail for {col}")
    idx = {s: i for i, s in enumerate(steps)}
    mat = np.zeros((len(active), len(steps)))
    for r, comp in enumerate(active):
        for s in steps:
            if comp in sets[s]:
                mat[r, idx[s]] = 1

    if order_by == "centroid":
        # temporal centre of mass of each row: succession reads as a staircase
        weights = np.arange(len(steps))
        centroid = (mat * weights).sum(axis=1) / np.maximum(mat.sum(axis=1), 1)
        order = np.argsort(centroid)
    elif order_by == "layer":
        order = np.argsort([sort_key(c) for c in active])
    else:
        raise ValueError(f"unknown order_by {order_by!r}")
    active = [active[i] for i in order]
    mat = mat[order]

    h = max(2.0, 0.115 * len(active) + 1.0)
    fig, ax = plt.subplots(figsize=(6.6, h))
    ax.imshow(np.ma.masked_where(mat == 0, mat), aspect="auto",
              cmap=mpl.colors.ListedColormap(["#1a6b3c"]), vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_facecolor("#fbfbfb")
    for edge in PHASE_EDGES:
        ax.axvline(np.searchsorted(steps, edge) - 0.5, color="#b03030", lw=0.9)
    for (lo, hi), name in zip(PHASE_SPANS, PHASE_NAMES):
        if not name:
            continue
        a = np.searchsorted(steps, lo) - 0.5
        b = min(np.searchsorted(steps, hi), len(steps)) - 0.5
        ax.text((a + b) / 2, -1.2, name, ha="center", va="bottom",
                fontsize=6.5, color="0.35")

    if order_by == "layer":
        layers = [parse_component(c)[0] for c in active]
        for i in range(1, len(active)):
            if layers[i] != layers[i - 1]:
                ax.axhline(i - 0.5, color="0.75", lw=0.5)

    ax.set_yticks(range(len(active)))
    ax.set_yticklabels(active, fontsize=5.6)
    for tick, name in zip(ax.get_yticklabels(), active):
        if parse_component(name)[2]:
            tick.set_fontweight("bold")
            tick.set_color("#333333")
        else:
            tick.set_color("#666666")

    tick_steps = [1, 1000, 10000, 40000, 80000, 143000]
    pos = [np.searchsorted(steps, t) for t in tick_steps if t <= steps[-1]]
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{t:,}" for t in tick_steps if t <= steps[-1]],
                       fontsize=6.5)
    ax.set_xlabel("Training step (checkpoints, not to scale)")
    ax.set_title(f"{title} — components above the {int(PERCENTILE * 100)}th "
                 f"percentile ($\\tau={TAU_MAIN}$)", fontsize=8, pad=20)
    ax.tick_params(length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 9 — role streams: who holds each role, with identity kept
# --------------------------------------------------------------------------

def _occupancy_share(steps, sets, window=7):
    """Per step, each upper-tail member contributes 1/|T_t|; then smoothed."""
    comps = sorted({c for s in steps for c in sets[s]}, key=sort_key)
    share = pd.DataFrame(0.0, index=steps, columns=comps)
    for s in steps:
        members = sets[s]
        if not members:
            continue
        for c in members:
            share.at[s, c] = 1.0 / len(members)
    return share.rolling(window, center=True, min_periods=1).mean()


def _shade_phases_indexed(ax, steps):
    """shade_phases, but in checkpoint-index space rather than step space --
    for figures (streams, heatmaps) whose x axis is position, not the step
    value itself."""
    for (lo, hi), colour in zip(PHASE_SPANS, PHASE_SHADE):
        a = np.searchsorted(steps, lo) - 0.5
        b = min(np.searchsorted(steps, hi), len(steps)) - 0.5
        if b <= a:
            continue
        ax.axvspan(a, b, color=colour, zorder=0, lw=0)


def fig_role_streams(dd, out, n_named=10, style="lanes"):
    """One lane per component (style='lanes') or a stacked area ('stack').

    Lanes are the default because in a stacked area the identity of a band has
    to be traced by colour, which fails once bands are thin or adjacent hues
    are close. In lanes each component owns a labelled horizontal strip, so no
    tracing is needed; the cost is that shares are no longer summed visually.
    """
    if style not in ("lanes", "stack"):
        raise ValueError(f"unknown streams style {style!r}")
    import matplotlib.patheffects as pe

    heights = {"lanes": 8.4, "stack": 7.2}[style]
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(6.4, heights),
                             gridspec_kw={"hspace": 0.30 if style == "lanes" else 0.20})
    tick_steps = [1, 1000, 10000, 40000, 80000, 143000]

    for ax, (col, title) in zip(axes, METRICS):
        steps, sets = upper_tail_sets(dd, col)
        x = np.arange(len(steps))
        _shade_phases_indexed(ax, steps)
        share = _occupancy_share(steps, sets)
        totals = share.sum(axis=0).sort_values(ascending=False)
        named = sorted(totals.index[:n_named], key=sort_key)
        rest = [c for c in share.columns if c not in named]
        palette = {c: component_colour(c) for c in named}

        if style == "stack":
            stacked = share[named].copy()
            if rest:
                stacked["other"] = share[rest].sum(axis=1)
            labels = named + (["other"] if rest else [])
            colours = [palette[c] for c in named] + \
                      ([np.array([0.82, 0.82, 0.82])] if rest else [])
            ax.stackplot(x, [stacked[c].to_numpy() for c in labels],
                         colors=colours, edgecolor="white", linewidth=0.15)
            cum = np.zeros(len(steps))
            for c in labels:
                v = stacked[c].to_numpy()
                j = int(np.argmax(v))
                if v[j] > 0.075:
                    ax.text(x[j], cum[j] + v[j] / 2, c, fontsize=5.4,
                            ha="center", va="center", zorder=6, color="0.05",
                            path_effects=[pe.withStroke(linewidth=1.5,
                                                        foreground="white")])
                cum = cum + v
            ax.set_ylim(0, 1)
            ax.set_ylabel(title.replace(" ", "\n", 1), fontsize=7.5)
        else:
            # "other" is excluded from the lanes: it is an aggregate of many
            # components, so its share sets a scale against which the named
            # lanes vanish. Its size is reported in the title instead.
            values = {c: share[c].to_numpy() for c in named}
            vmax = max(v.max() for v in values.values())
            if vmax <= 0:
                raise ValueError(f"no occupancy recorded for {col}")
            coverage = share[named].to_numpy().sum() / share.to_numpy().sum()
            for i, comp in enumerate(reversed(named)):
                v = values[comp] / vmax * 0.90
                ax.fill_between(x, i, i + v, color=palette[comp], lw=0.4,
                                edgecolor="white", zorder=3)
                ax.axhline(i, color="0.88", lw=0.4, zorder=1)
            ax.set_ylim(-0.15, len(named))
            ax.set_yticks([i + 0.35 for i in range(len(named))])
            ax.set_yticklabels(list(reversed(named)), fontsize=6.2)
            for tick, comp in zip(ax.get_yticklabels(), reversed(named)):
                if parse_component(comp)[2]:
                    tick.set_fontweight("bold")
            ax.set_title(f"{title}  —  these {len(named)} components account for "
                         f"{coverage:.0%} of upper-tail membership;\n"
                         f"full lane $=$ {vmax:.2f} share", fontsize=7.5,
                         pad=3, loc="left")

        for edge in PHASE_EDGES:
            ax.axvline(np.searchsorted(steps, edge) - 0.5, color="0.15",
                       lw=0.8, ls=(0, (4, 2)), zorder=5)
        ax.set_xlim(0, len(steps) - 1)
        ax.grid(False)
        pos = [np.searchsorted(steps, t) for t in tick_steps if t <= steps[-1]]
        ax.set_xticks(pos)
        if ax is axes[-1]:
            ax.set_xticklabels([f"{t:,}" for t in tick_steps if t <= steps[-1]],
                               fontsize=6.5)
            ax.set_xlabel("Training step (checkpoints, not to scale)")
        else:
            ax.tick_params(labelbottom=False)
        ax.tick_params(length=2)

    if style == "stack":
        axes[0].set_title("Share of the upper tail held by each component "
                          f"($\\tau={TAU_MAIN}$, {int(PERCENTILE*100)}th percentile)",
                          fontsize=8, pad=6)
    bottom_legends(fig, y=-0.006 if style == "lanes" else -0.012)
    fig.savefig(os.path.join(out, "role_streams.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 10 — role profile: which components hold which role, by phase
# --------------------------------------------------------------------------

PHASE_COLUMNS = [(1000, 10000, "expl."), (10000, 80000, "cons."),
                 (80000, 10 ** 9, "refin.")]


def fig_role_profile(dd, out, n_per_metric=10, min_share=0.10):
    per_metric = {}
    for col, _ in METRICS:
        steps, sets = upper_tail_sets(dd, col)
        per_metric[col] = (steps, sets)

    keep = set()
    for col, _ in METRICS:
        steps, sets = per_metric[col]
        occ = {}
        for s in steps:
            for c in sets[s]:
                occ[c] = occ.get(c, 0) + 1
        keep |= set(sorted(occ, key=occ.get, reverse=True)[:n_per_metric])
    rows = sorted(keep, key=sort_key)

    blocks = []
    for col, _ in METRICS:
        steps, sets = per_metric[col]
        block = np.zeros((len(rows), len(PHASE_COLUMNS)))
        for k, (lo, hi, _) in enumerate(PHASE_COLUMNS):
            in_phase = [s for s in steps if lo <= s < hi]
            if not in_phase:
                raise ValueError(f"phase [{lo},{hi}) contains no checkpoints")
            for r, comp in enumerate(rows):
                block[r, k] = sum(comp in sets[s] for s in in_phase) / len(in_phase)
        blocks.append(block)
    mat = np.concatenate(blocks, axis=1)

    fig, ax = plt.subplots(figsize=(5.6, 0.145 * len(rows) + 1.4))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#f4f4f4")
    im = ax.imshow(np.ma.masked_where(mat < min_share, mat), aspect="auto",
                   cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    for g in range(1, len(METRICS)):
        ax.axvline(g * len(PHASE_COLUMNS) - 0.5, color="0.2", lw=1.1)
    layers = [parse_component(c)[0] for c in rows]
    for i in range(1, len(rows)):
        if layers[i] != layers[i - 1]:
            ax.axhline(i - 0.5, color="0.75", lw=0.5)

    ax.set_xticks(range(len(METRICS) * len(PHASE_COLUMNS)))
    ax.set_xticklabels([lab for _ in METRICS for *_, lab in PHASE_COLUMNS],
                       fontsize=6, rotation=90)
    for g, (_, title) in enumerate(METRICS):
        ax.text(g * len(PHASE_COLUMNS) + 1, -1.1, title.replace(" ", "\n"),
                ha="center", va="bottom", fontsize=6.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=5.8)
    for tick, name in zip(ax.get_yticklabels(), rows):
        if parse_component(name)[2]:
            tick.set_fontweight("bold")
    ax.tick_params(length=2)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="fraction of the phase spent in the upper tail")
    fig.savefig(os.path.join(out, "role_profile.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------

BUILDERS = {
    "topology": lambda dm, dd, de, out: fig_topology(dm, out),
    "distributions": lambda dm, dd, de, out: fig_distributions(dd, de, out),
    "centrality": lambda dm, dd, de, out: fig_centrality_distributions(dd, out),
    "turnover": lambda dm, dd, de, out: fig_role_turnover(dd, out),
    "regimes": lambda dm, dd, de, out: fig_regime_similarity(dd, out),
    "layers": lambda dm, dd, de, out: fig_layer_occupancy(dd, out),
    "traces": lambda dm, dd, de, out: fig_component_traces(dd, out),
    "streams": lambda dm, dd, de, out: fig_role_streams(dd, out,
                                                        style=STREAMS_STYLE),
    "profile": lambda dm, dd, de, out: fig_role_profile(dd, out),
    "heatmaps": lambda dm, dd, de, out: [
        fig_heatmap(dd, col, title.capitalize(),
                    f"heatmap_{col.replace('_dist', '')}.pdf", out)
        for col, title in METRICS
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=".", help="directory holding the pickles")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--only", nargs="+", choices=sorted(BUILDERS),
                    help="build only these figures (default: all)")
    ap.add_argument("--order-by", default="centroid", choices=["centroid", "layer"],
                    help="row ordering for the appendix heatmaps")
    ap.add_argument("--streams-style", default="lanes", choices=["lanes", "stack"],
                    help="role_streams as labelled lanes or as a stacked area")
    args = ap.parse_args()

    global STREAMS_STYLE
    STREAMS_STYLE = args.streams_style

    os.makedirs(args.out, exist_ok=True)
    mpl.rcParams.update(RC)
    dm, dd, de = load(args.data)
    print(f"loaded {dd.step.nunique()} checkpoints, "
          f"{dd.component.nunique()} components, "
          f"thresholds {sorted(dm.threshold.unique())}")

    if args.order_by == "layer":
        BUILDERS["heatmaps"] = lambda dm, dd, de, out: [
            fig_heatmap(dd, col, title.capitalize(),
                        f"heatmap_{col.replace('_dist', '')}.pdf", out,
                        order_by="layer")
            for col, title in METRICS
        ]

    for name in (args.only or sorted(BUILDERS)):
        BUILDERS[name](dm, dd, de, args.out)
        print(f"  wrote {name}")


if __name__ == "__main__":
    main()