"""Pure Plotly figure/table builders for the dashboard (no I/O, no model code).

Conventions kept visible: unsmoothed values, behavioral and graph quantities in
separate panels with their own units, step 0 drawn as a labeled initialization
marker on log axes, and display limits that never alter full-graph metrics.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nnets_transformer.analysis.longitudinal import checkpoint_similarity

INIT_LABEL = "step 0 (initialization)"


def axis_positions(values, log_x):
    """Map training-axis values to plotted x; step/tokens 0 sits left of the first positive value on log axes."""
    values = [float(v) for v in values]
    if not log_x:
        return values, None
    positive = [v for v in values if v > 0]
    if not positive:
        return [1.0 for _ in values], 1.0
    zero_at = min(positive) / 4
    return [v if v > 0 else zero_at for v in values], (zero_at if any(v <= 0 for v in values) else None)


def training_axis(frame, axis):
    if axis == "tokens":
        if "tokens_seen" not in frame or frame.tokens_seen.isna().any():
            raise ValueError("tokens seen are unavailable for this trajectory")
        return frame.tokens_seen.astype(float), "Training tokens seen"
    return frame.step.astype(float), "Training step"


def _style_axis(figure, log_x, zero_at, title, rows):
    for row in range(1, rows+1):
        figure.update_xaxes(type="log" if log_x else "linear", row=row, col=1, showgrid=True)
    figure.update_xaxes(title_text=title + (" (log scale)" if log_x else ""), row=rows, col=1)
    if zero_at is not None:
        figure.add_vline(x=zero_at, line_dash="dot", line_color="#888", annotation_text=INIT_LABEL,
                         annotation_position="top right", annotation_font_size=10)


def checkpoint_summary(checkpoints, threshold):
    """One row per checkpoint at the threshold; behavior is threshold-independent and deduplicated."""
    rows = checkpoints[checkpoints.threshold == threshold].sort_values("step")
    if rows.step.duplicated().any():
        raise ValueError("Duplicate checkpoint rows for one trajectory/threshold")
    return rows


def overview(checkpoints, threshold, axis="step", log_x=True, selected_step=None, behavior_label="Clean behavioral score"):
    rows = checkpoint_summary(checkpoints, threshold)
    x_raw, title = training_axis(rows, axis)
    x, zero_at = axis_positions(x_raw, log_x)
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.04,
        subplot_titles=("Nodes (complete inventory vs active)", "Edges", "Density", behavior_label))
    custom = np.stack([rows.step, rows.checkpoint.astype(str)], axis=-1)
    hover = "step %{customdata[0]} (%{customdata[1]})<br>%{y}<extra>%{fullData.name}</extra>"
    def trace(column, name, row, **style):
        if column in rows and rows[column].notna().any():
            figure.add_trace(go.Scatter(x=x, y=rows[column], name=name, mode="lines+markers", customdata=custom,
                                        hovertemplate=hover, **style), row=row, col=1)
    trace("num_nodes", "nodes (inventory)", 1)
    trace("active_nodes", "active nodes", 1, line_dash="dash")
    trace("num_edges", "edges", 2)
    trace("density", "density", 3)
    trace("admissible_density", "admissible-pair density", 3, line_dash="dash")
    trace("clean_score", "clean score", 4)
    trace("legacy_induction_loss", "legacy induction loss", 4, line_dash="dot")
    if selected_step is not None and selected_step in set(rows.step):
        position = x[list(rows.step).index(selected_step)]
        figure.add_vline(x=position, line_color="#d62728", line_width=1.5)
    _style_axis(figure, log_x, zero_at, title, 4)
    figure.update_layout(height=820, margin=dict(l=60, r=20, t=40, b=40), hovermode="closest",
                         legend=dict(orientation="h", y=-.06), clickmode="event+select")
    return figure


def ranking(components, step, metric, top=25):
    rows = components[(components.step == step) & (components.metric == metric)][["component", "layer", "head", "value", "units"]]
    rows = rows.sort_values(["value", "component"], ascending=[False, True]).reset_index(drop=True)
    # Ties share the minimum rank.
    rows["rank"] = rows.value.rank(ascending=False, method="min").astype("Int64")
    return rows.head(top) if top else rows


def component_trajectories(components, metric, selected, axis="step", log_x=True, ranks=False, selected_step=None):
    rows = components[(components.metric == metric)].copy()
    if ranks:
        rows["value"] = rows.groupby("step").value.rank(ascending=False, method="min")
    figure = go.Figure()
    zero_at, title = None, "Training step"
    for component in selected:
        series = rows[rows.component == component].sort_values("step")
        if series.empty:
            continue
        x_raw, title = training_axis(series, axis)
        x, zero_here = axis_positions(x_raw, log_x)
        zero_at = zero_at or zero_here
        figure.add_trace(go.Scatter(x=x, y=series.value, mode="lines+markers", name=component,
                                    customdata=series.step, hovertemplate="step %{customdata}<br>%{y}<extra>%{fullData.name}</extra>"))
    figure.update_xaxes(type="log" if log_x else "linear", title_text=title + (" (log scale)" if log_x else ""))
    if zero_at is not None:
        figure.add_vline(x=zero_at, line_dash="dot", line_color="#888", annotation_text=INIT_LABEL)
    figure.update_yaxes(title_text=f"{metric} rank (1 = highest)" if ranks else metric, autorange="reversed" if ranks else True)
    if not selected:
        figure.add_annotation(text="Pin components to follow them over training", showarrow=False, x=.5, y=.5, xref="paper", yref="paper")
    figure.update_layout(height=380, margin=dict(l=60, r=20, t=30, b=40))
    return figure


def heatmap(components, metric, top=30):
    """Metric-by-step heatmap for the components most often in the top ranks."""
    rows = components[components.metric == metric]
    if rows.empty:
        return go.Figure()
    pivot = rows.pivot_table(index="component", columns="step", values="value", aggfunc="first")
    order = pivot.mean(axis=1).sort_values(ascending=False).index[:top]
    pivot = pivot.loc[order]
    figure = go.Figure(go.Heatmap(z=pivot.values, x=[str(s) for s in pivot.columns], y=list(pivot.index),
                                  colorscale="Viridis", colorbar=dict(title=metric),
                                  hovertemplate="step %{x}<br>%{y}<br>%{z}<extra></extra>"))
    figure.update_xaxes(type="category", title_text="Training step (recorded checkpoints, categorical)")
    figure.update_layout(height=max(320, 16*len(pivot)+120), margin=dict(l=160, r=20, t=30, b=60))
    return figure


def distribution(components, steps, metric, mode="histogram", bins=30):
    rows = components[(components.metric == metric) & components.step.isin(steps)]
    figure = go.Figure()
    if rows.empty:
        return figure, pd.DataFrame()
    lo, hi = float(rows.value.min()), float(rows.value.max())
    size = (hi-lo)/bins if hi > lo else 1.0
    summary = []
    for step in steps:
        values = rows[rows.step == step].value.dropna().sort_values()
        if values.empty:
            continue
        if mode == "ecdf":
            figure.add_trace(go.Scatter(x=values, y=np.arange(1, len(values)+1)/len(values), mode="lines", name=f"step {step}", line_shape="hv"))
        else:
            # Shared bins across checkpoints keep shapes comparable.
            figure.add_trace(go.Histogram(x=values, name=f"step {step}", opacity=.6, xbins=dict(start=lo, end=hi+size, size=size)))
        summary.append({"step": step, "n": len(values), "min": values.min(), "q25": values.quantile(.25),
                        "median": values.median(), "q75": values.quantile(.75), "q95": values.quantile(.95), "max": values.max()})
    figure.update_layout(barmode="overlay", height=360, margin=dict(l=60, r=20, t=30, b=40),
                         xaxis_title=metric, yaxis_title="fraction of components" if mode == "ecdf" else "components")
    return figure, pd.DataFrame(summary)


def roles(role_rows, metric, axis_steps=None, log_x=True):
    rows = role_rows[role_rows.metric == metric].sort_values("step")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Upper-tail set size", "Turnover vs previous checkpoint / Jaccard with final set"))
    if rows.empty:
        return figure
    x, zero_at = axis_positions(rows.step, log_x)
    figure.add_trace(go.Scatter(x=x, y=rows["size"], mode="lines+markers", name="members", customdata=rows.step,
                                hovertemplate="step %{customdata}: %{y}<extra></extra>"), row=1, col=1)
    figure.add_trace(go.Scatter(x=x, y=rows.churn, mode="lines+markers", name="churn (1 - Jaccard to previous)", customdata=rows.step), row=2, col=1)
    figure.add_trace(go.Scatter(x=x, y=rows.final_jaccard, mode="lines+markers", name="Jaccard with final checkpoint", customdata=rows.step), row=2, col=1)
    _style_axis(figure, log_x, zero_at, "Training step", 2)
    figure.update_layout(height=520, margin=dict(l=60, r=20, t=40, b=40), legend=dict(orientation="h", y=-.12))
    return figure


def membership_change(role_rows, metric, step):
    rows = role_rows[role_rows.metric == metric].sort_values("step")
    steps = list(rows.step)
    if step not in steps:
        return {"entering": [], "leaving": [], "retained": [], "previous_step": None}
    index = steps.index(step)
    current = set(rows.iloc[index].members)
    previous = set(rows.iloc[index-1].members) if index else set()
    return {"entering": sorted(current-previous), "leaving": sorted(previous-current), "retained": sorted(current & previous),
            "previous_step": int(steps[index-1]) if index else None}


def similarity_matrix(components, metric, quantile):
    steps, matrix = checkpoint_similarity(components, metric, quantile)
    labels = [str(s) for s in steps]
    z = [[math.nan if v is None else v for v in row] for row in matrix]
    figure = go.Figure(go.Heatmap(z=z, x=labels, y=labels, zmin=0, zmax=1, colorscale="Blues",
                                  colorbar=dict(title="Jaccard"), hovertemplate="%{y} vs %{x}: %{z}<extra></extra>"))
    figure.update_xaxes(type="category", title_text="step")
    figure.update_yaxes(type="category", title_text="step")
    figure.update_layout(height=420, margin=dict(l=60, r=20, t=30, b=40))
    return figure


def compare(series, column, axis="step", log_x=True, label=None):
    """Overlay one graph quantity across compatible trajectories (dict label -> checkpoint rows)."""
    figure = go.Figure()
    zero_at = None
    title = "Training step"
    for name, rows in series.items():
        if column not in rows or rows[column].isna().all():
            continue
        x_raw, title = training_axis(rows, axis)
        x, zero_here = axis_positions(x_raw, log_x)
        zero_at = zero_at or zero_here
        figure.add_trace(go.Scatter(x=x, y=rows[column], mode="lines+markers", name=name, customdata=rows.step,
                                    hovertemplate="step %{customdata}<br>%{y}<extra>%{fullData.name}</extra>"))
    figure.update_xaxes(type="log" if log_x else "linear", title_text=title + (" (log scale)" if log_x else ""))
    if zero_at is not None:
        figure.add_vline(x=zero_at, line_dash="dot", line_color="#888", annotation_text=INIT_LABEL)
    figure.update_layout(height=380, yaxis_title=label or column, margin=dict(l=60, r=20, t=30, b=40), legend=dict(orientation="h", y=-.2))
    return figure


def behavior_facets(series_by_task, axis="step", log_x=True):
    """Behavioral scores per task in separate panels: units differ across tasks."""
    tasks = sorted(series_by_task)
    figure = make_subplots(rows=max(1, len(tasks)), cols=1, shared_xaxes=False,
                           subplot_titles=[f"{task} ({units})" for task, (units, _) in sorted(series_by_task.items())] or [""])
    for row, task in enumerate(tasks, 1):
        units, series = series_by_task[task]
        for name, rows in series.items():
            x_raw, _ = training_axis(rows, axis)
            x, _ = axis_positions(x_raw, log_x)
            figure.add_trace(go.Scatter(x=x, y=rows.clean_score, mode="lines+markers", name=f"{task}: {name}"), row=row, col=1)
        figure.update_xaxes(type="log" if log_x else "linear", row=row, col=1)
    figure.update_layout(height=300*max(1, len(tasks)), margin=dict(l=60, r=20, t=40, b=40))
    return figure


# -- network ------------------------------------------------------------------
SITE_ORDER = {"embed": 0, "pos_embed": 1, "q": 2, "k": 3, "v": 4, "rot_q": 5, "rot_k": 6, "z": 7, "mlp": 8, "logits": 9}


def _site(component):
    if "embed" in component or component in {"emb", "pos_emb"}:
        return "pos_embed" if "pos" in component else "embed"
    if "mlp" in component:
        return "mlp"
    if component == "logits":
        return "logits"
    for site in ("rot_q", "rot_k", "q", "k", "v", "z"):
        if f"hook_{site}" in component or f"attn.{site}." in component:
            return site
    return "other"


def layout(inventory):
    """Deterministic positions from the complete inventory: columns by layer, rows by site/head."""
    rows = inventory.drop_duplicates("component").copy()
    rows["site"] = rows.component.map(_site)
    rows["order"] = rows.site.map(lambda s: SITE_ORDER.get(s, 99))
    rows["head_order"] = pd.to_numeric(rows["head"], errors="coerce").fillna(-1)
    rows = rows.sort_values(["layer", "order", "head_order", "component"])
    positions = {}
    for layer, group in rows.groupby("layer", sort=True):
        for index, component in enumerate(group.component):
            positions[component] = {"x": 220*(int(layer)+1), "y": 34*index}
    return positions


def network_elements(nodes, edges, positions, metric_values=None, selected=None, neighborhood=False,
                     min_strength=None, max_edges=300):
    """Display subset of a full graph. Returns (elements, counts); full-graph metrics are not recomputed."""
    total_edges = len(edges)
    shown = edges
    if min_strength is not None:
        shown = shown[shown.strength >= min_strength]
    if neighborhood and selected:
        shown = shown[(shown.source == selected) | (shown.target == selected)]
    shown = shown.sort_values(["strength", "source", "target"], ascending=[False, True, True])
    filtered = len(shown)
    if max_edges is not None:
        shown = shown.head(max_edges)
    visible_nodes = list(nodes)
    if neighborhood and selected:
        visible_nodes = sorted({selected, *shown.source, *shown.target} & set(nodes))
    values = metric_values or {}
    top = max([abs(v) for v in values.values() if v is not None and not math.isnan(v)] or [1]) or 1
    elements = []
    for node in visible_nodes:
        value = values.get(node)
        size = 14 + 26*(abs(value)/top if value is not None and not math.isnan(value) else 0)
        elements.append({"data": {"id": node, "label": node.replace("blocks.", "").replace("hook_", ""), "value": value,
                                  "size": size}, "position": positions.get(node, {"x": 0, "y": 0}),
                         "classes": "selected" if node == selected else ""})
    for row in shown.itertuples():
        elements.append({"data": {"id": f"{row.source}->{row.target}", "source": row.source, "target": row.target,
                                  "strength": float(row.strength), "width": 1+4*min(1., float(row.strength)/2)},
                         "classes": getattr(row, "change", "") or ""})
    counts = {"nodes_total": len(nodes), "nodes_displayed": len(visible_nodes), "edges_total": total_edges,
              "edges_after_filters": filtered, "edges_displayed": len(shown),
              "truncated": len(shown) < total_edges}
    return elements, counts


def edge_difference(first, second):
    """Edges at checkpoint A versus B, keyed by directed endpoint identity."""
    a = first.set_index(["source", "target"]).strength
    b = second.set_index(["source", "target"]).strength
    keys = sorted(set(a.index) | set(b.index))
    rows = []
    for key in keys:
        in_a, in_b = key in a.index, key in b.index
        rows.append({"source": key[0], "target": key[1], "strength_a": a.get(key) if in_a else None,
                     "strength_b": b.get(key) if in_b else None,
                     "change": "both" if in_a and in_b else "only_a" if in_a else "only_b",
                     "strength": float(b.get(key) if in_b else a.get(key))})
    return pd.DataFrame(rows, columns=["source", "target", "strength_a", "strength_b", "change", "strength"])
