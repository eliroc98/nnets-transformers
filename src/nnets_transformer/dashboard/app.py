"""Dash application factory. Nothing is loaded at import time; data access is CPU-only.

Shared selection (export, trajectory, threshold, metric, checkpoint, pinned
components) drives every view. Display limits apply to the browser view only.
"""
import base64
import functools
import json
import time
import traceback

import dash
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update
import dash_cytoscape as cyto
import pandas as pd

from . import figures
from .exports import chart_html, provenance, table_bytes
from .state import ViewState, reconcile
from .status import attempts, tail

TABS = ("overview", "components", "distributions", "roles", "compare", "network", "status")


def _options(values, label=str):
    return [{"label": label(v), "value": v} for v in values]


def _trajectory_label(row):
    parts = [row.get("model_id"), row.get("task"), row.get("replacement"), row.get("graph_version"),
             row.get("resolution"), row.get("dtype"), row.get("training_seed")]
    return " · ".join(str(p) for p in parts if p is not None and p == p) + f" [{row['checkpoints']} ckpts]"


def create_app(repository, config, job=None):
    app = dash.Dash(__name__, title="nnets-transformer training dashboard", suppress_callback_exceptions=True)
    app.repository = repository
    timings = app.timings = {}

    def logged(name):
        def wrap(function):
            @functools.wraps(function)
            def inner(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                except dash.exceptions.PreventUpdate:
                    raise
                except Exception as exc:
                    if job is not None:
                        job.event("callback_error", severity="ERROR", callback=name, message=str(exc),
                                  traceback=traceback.format_exc())
                    raise
                finally:
                    timings.setdefault(name, []).append(time.perf_counter()-started)
                    del timings[name][:-200]
            return inner
        return wrap

    def current(export_id, trajectory_id, threshold, metric, step_index, pinned=None, axis="step", log_x=None, quantile=.95):
        state = ViewState(export_id=export_id, trajectory_id=trajectory_id, threshold=threshold, metric=metric,
                          pinned=pinned or [], axis=axis or "step", log_x=bool(log_x and "log" in log_x), quantile=quantile or .95)
        if export_id and trajectory_id:
            steps = repository.steps(export_id, trajectory_id)
            if step_index is not None and 0 <= step_index < len(steps):
                state.step = steps[step_index]
        return state

    control = {"display": "inline-block", "marginRight": "12px", "verticalAlign": "top"}
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="requested-view"),
        dcc.Download(id="download"),
        html.H3("Component-influence graphs over training"),
        html.Div([
            html.Div([html.Label("Export"), dcc.Dropdown(id="export", clearable=False, style={"width": "330px"})], style=control),
            html.Div([html.Label("Trajectory (model · task · replacement · graph)"),
                      dcc.Dropdown(id="trajectory", clearable=False, style={"width": "560px"})], style=control),
            html.Div([html.Label("Threshold"), dcc.Dropdown(id="threshold", clearable=False, style={"width": "110px"})], style=control),
            html.Div([html.Label("Component metric"), dcc.Dropdown(id="metric", clearable=False, style={"width": "170px"})], style=control),
            html.Div([html.Label("Training axis"), dcc.RadioItems(id="axis", options=[{"label": "steps", "value": "step"},
                      {"label": "tokens seen", "value": "tokens"}], value="step", inline=True),
                      dcc.Checklist(id="log-x", options=[{"label": "log x (step 0 marked)", "value": "log"}], value=["log"])], style=control),
            html.Div([html.Button("Refresh artifacts", id="refresh"), html.Br(),
                      dcc.Upload(html.Button("Load saved view"), id="upload-view")], style=control),
        ]),
        html.Div(id="catalog-notes", style={"color": "#a33", "whiteSpace": "pre-wrap", "fontSize": "13px"}),
        html.Div([html.Button("◀ previous", id="previous"), html.Button("next ▶", id="next"),
                  html.Span(id="checkpoint-label", style={"marginLeft": "12px", "fontWeight": "bold"}),
                  html.Div(dcc.Slider(id="checkpoint", min=0, max=0, step=1, value=0, marks=None,
                           tooltip={"placement": "bottom"}), style={"width": "70%", "display": "inline-block", "verticalAlign": "middle"})]),
        html.Div(id="provenance", style={"fontSize": "12px", "color": "#555", "whiteSpace": "pre-wrap"}),
        html.Div([html.Button("Export view (JSON)", id="export-view"), html.Button("Export overview chart (HTML)", id="export-chart"),
                  html.Button("Export ranking (CSV)", id="export-ranking"), html.Button("Export checkpoint table (Parquet)", id="export-table"),
                  html.Span(" Share: "), dcc.Input(id="share-link", readOnly=True, style={"width": "640px"})]),
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview", children=[
                html.Div([dcc.Graph(id="overview-graph")], style={"width": "68%", "display": "inline-block"}),
                html.Div(id="checkpoint-summary", style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "fontSize": "13px"})]),
            dcc.Tab(label="Components", value="components", children=[
                html.Div([html.Label("Pinned components (click a ranking row to add)"),
                          dcc.Dropdown(id="pinned", multi=True), dcc.Checklist(id="rank-mode", options=[{"label": "show ranks", "value": "rank"}], value=[])]),
                html.Div([dash_table.DataTable(id="ranking", page_size=25, sort_action="native",
                                               style_table={"overflowX": "auto"}, style_cell={"fontSize": "12px"})],
                         style={"width": "38%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div([dcc.Graph(id="trajectories"), dcc.Graph(id="heatmap")], style={"width": "60%", "display": "inline-block"})]),
            dcc.Tab(label="Distributions", value="distributions", children=[
                dcc.RadioItems(id="dist-mode", options=_options(["histogram", "ecdf"]), value="histogram", inline=True),
                html.Label("Additional checkpoints"), dcc.Dropdown(id="dist-steps", multi=True),
                dcc.Graph(id="distribution"), dash_table.DataTable(id="dist-summary", style_cell={"fontSize": "12px"})]),
            dcc.Tab(label="Role dynamics", value="roles", children=[
                html.Div([html.Label("Upper-tail quantile (strictly above; ties excluded)"),
                          dcc.Slider(id="quantile", min=.5, max=.99, step=.01, value=.95, marks={.5: ".50", .9: ".90", .95: ".95", .99: ".99"})],
                         style={"width": "40%"}),
                dcc.Graph(id="roles-graph"), html.Div(id="membership", style={"fontSize": "13px", "whiteSpace": "pre-wrap"}),
                dcc.Graph(id="similarity")]),
            dcc.Tab(label="Compare runs", value="compare", children=[
                html.Label("Comparison trajectories (same export)"), dcc.Dropdown(id="compare", multi=True),
                dcc.Dropdown(id="compare-column", value="density", clearable=False, style={"width": "280px"},
                             options=_options(["num_nodes", "active_nodes", "num_edges", "density", "admissible_density"])),
                html.Div(id="compare-notes", style={"fontSize": "13px", "color": "#a33", "whiteSpace": "pre-wrap"}),
                dcc.Graph(id="compare-graph"), dcc.Graph(id="behavior-facets")]),
            dcc.Tab(label="Network", value="network", children=[
                html.Div([html.Label("Minimum strength"), dcc.Input(id="min-strength", type="number", min=0, step=.01, debounce=True),
                          html.Label(" Max displayed edges"), dcc.Input(id="max-edges", type="number", min=1, value=config.get("network_max_edges", 300), debounce=True),
                          dcc.Checklist(id="neighborhood", options=[{"label": "neighborhood of selected node only", "value": "on"}], value=[], inline=True),
                          html.Label(" Difference against step"), dcc.Dropdown(id="compare-step", style={"width": "160px", "display": "inline-block"})]),
                html.Div(id="network-counts", style={"fontSize": "13px", "fontWeight": "bold"}),
                cyto.Cytoscape(id="network", layout={"name": "preset"}, style={"width": "100%", "height": "620px"}, elements=[],
                               stylesheet=[{"selector": "node", "style": {"label": "data(label)", "width": "data(size)", "height": "data(size)", "font-size": "8px", "background-color": "#4c78a8"}},
                                           {"selector": "edge", "style": {"width": "data(width)", "curve-style": "bezier", "target-arrow-shape": "triangle", "opacity": .6}},
                                           {"selector": ".only_a", "style": {"line-color": "#d62728", "target-arrow-color": "#d62728"}},
                                           {"selector": ".only_b", "style": {"line-color": "#2ca02c", "target-arrow-color": "#2ca02c"}},
                                           {"selector": ".selected", "style": {"background-color": "#ff7f0e"}}]),
                html.Div(id="node-details", style={"fontSize": "13px", "whiteSpace": "pre-wrap"})]),
            dcc.Tab(label="Run status / logs", value="status", children=[
                dcc.Interval(id="status-interval", interval=int(config.get("status_refresh_seconds", 15)*1000)),
                dash_table.DataTable(id="status-table", row_selectable="single", page_size=15, style_cell={"fontSize": "12px", "textAlign": "left"},
                                     columns=[{"name": c, "id": c} for c in ("job_id", "attempt_id", "state", "phase", "liveness", "counters", "updated_at", "last_progress_at", "exit_code")]),
                dcc.Dropdown(id="severity", options=_options(["INFO", "WARNING", "ERROR"]), placeholder="severity filter", style={"width": "200px"}),
                html.Pre(id="log-tail", style={"maxHeight": "420px", "overflowY": "scroll", "fontSize": "11px", "background": "#f7f7f7"})]),
        ]),
    ], style={"fontFamily": "sans-serif", "margin": "12px"})

    # -- requested view (URL or uploaded JSON) -------------------------------
    @app.callback(Output("requested-view", "data"), Input("url", "search"), Input("upload-view", "contents"))
    @logged("requested_view")
    def requested(search, uploaded):
        try:
            if ctx.triggered_id == "upload-view" and uploaded:
                state = ViewState.from_json(base64.b64decode(uploaded.split(",", 1)[1]).decode())
                source = "saved view"
            else:
                state = ViewState.from_query(search)
                source = "URL"
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return {"error": f"Could not read requested view: {exc}"}
        return {"state": json.loads(state.to_json())["state"], "source": source}

    def requested_state(data):
        if not data or "state" not in data:
            return ViewState()
        return ViewState.from_dict({"state": data["state"]})

    @app.callback(Output("export", "options"), Output("export", "value"), Output("catalog-notes", "children"),
                  Input("requested-view", "data"), Input("refresh", "n_clicks"), State("export", "value"))
    @logged("catalog")
    def catalog(data, _refresh, export_value):
        if ctx.triggered_id == "refresh":
            repository.refresh()
            if job is not None:
                job.event("artifacts_refreshed", exports=len(repository.catalog))
        notes = [] if not data or "error" not in data else [data["error"]]
        state = requested_state(data)
        if ctx.triggered_id == "refresh" and export_value:
            state.export_id = export_value
        state, reconciled = reconcile(state, repository)
        notes += reconciled + [f"Skipped {p['path']}: {p['problem']}" for p in repository.problems]
        options = [{"label": f"{e['name']}{' (legacy)' if e['legacy'] else ''}", "value": key} for key, e in repository.catalog.items()]
        return options, state.export_id, "\n".join(notes)

    @app.callback(Output("trajectory", "options"), Output("trajectory", "value"), Input("export", "value"), State("requested-view", "data"))
    @logged("trajectories")
    def trajectories(export_id, data):
        if not export_id:
            return [], None
        frame = repository.trajectories(export_id)
        state = requested_state(data)
        state.export_id = export_id
        state, _ = reconcile(state, repository)
        return [{"label": _trajectory_label(r), "value": r["trajectory_id"]} for r in frame.to_dict("records")], state.trajectory_id

    @app.callback(Output("threshold", "options"), Output("threshold", "value"), Output("metric", "options"), Output("metric", "value"),
                  Output("checkpoint", "max"), Output("checkpoint", "marks"), Output("pinned", "options"),
                  Output("compare", "options"), Output("dist-steps", "options"), Output("compare-step", "options"),
                  Output("axis", "value"), Output("log-x", "value"), Output("provenance", "children"),
                  Input("trajectory", "value"), State("export", "value"), State("requested-view", "data"))
    @logged("trajectory_controls")
    def controls(trajectory_id, export_id, data):
        if not export_id or not trajectory_id:
            return [], None, [], None, 0, {}, [], [], [], [], "step", ["log"], ""
        state = requested_state(data)
        state.export_id, state.trajectory_id = export_id, trajectory_id
        state, _ = reconcile(state, repository)
        steps = repository.steps(export_id, trajectory_id)
        marks = {i: str(s) for i, s in enumerate(steps) if len(steps) <= 16 or i % max(1, len(steps)//12) == 0 or i == len(steps)-1}
        components = sorted(repository.select(export_id, "components", trajectory_id).component.unique())
        frame = repository.trajectories(export_id)
        others = [{"label": _trajectory_label(r), "value": r["trajectory_id"]} for r in frame.to_dict("records") if r["trajectory_id"] != trajectory_id]
        entry = repository.entry(export_id)
        rows = repository.select(export_id, "checkpoints", trajectory_id)
        info = [f"Export {entry['name']} · manifest {entry['manifest_sha256'][:12]} · graph {entry['manifest'].get('graph', {}).get('version')}"
                f" · scoring {rows.scoring_version.iloc[0]} · measurement {rows.measurement_status.iloc[0]}"
                f" (coverage {rows.measurement_coverage.iloc[0]})"]
        info += [f"Limitation: {l}" for l in entry["limitations"]]
        return (_options(repository.thresholds(export_id, trajectory_id)), state.threshold,
                _options(repository.metrics(export_id, trajectory_id)), state.metric,
                max(0, len(steps)-1), marks, _options(components), others, _options(steps), _options(steps),
                state.axis, ["log"] if state.log_x else [], "\n".join(info))

    @app.callback(Output("checkpoint", "value"), Input("overview-graph", "clickData"), Input("previous", "n_clicks"),
                  Input("next", "n_clicks"), Input("checkpoint", "max"), State("checkpoint", "value"),
                  State("trajectory", "value"), State("export", "value"), State("requested-view", "data"))
    @logged("checkpoint_selection")
    def select_checkpoint(click, _previous, _next, maximum, value, trajectory_id, export_id, data):
        if not export_id or not trajectory_id:
            return 0
        steps = repository.steps(export_id, trajectory_id)
        value = value or 0
        if ctx.triggered_id == "previous":
            return max(0, value-1)
        if ctx.triggered_id == "next":
            return min(len(steps)-1, value+1)
        if ctx.triggered_id == "overview-graph" and click:
            step = int(click["points"][0]["customdata"][0])
            return steps.index(step) if step in steps else no_update
        state = requested_state(data)
        state.export_id, state.trajectory_id = export_id, trajectory_id
        state, _ = reconcile(state, repository)
        return steps.index(state.step) if state.step in steps else len(steps)-1

    @app.callback(Output("pinned", "value"), Input("ranking", "active_cell"), Input("trajectory", "value"),
                  State("ranking", "data"), State("pinned", "value"), State("export", "value"), State("requested-view", "data"))
    @logged("pinning")
    def pin(cell, trajectory_id, rows, pinned, export_id, data):
        if ctx.triggered_id == "ranking" and cell and rows:
            component = rows[cell["row"]]["component"]
            return list(dict.fromkeys([*(pinned or []), component]))
        state = requested_state(data)
        state.export_id, state.trajectory_id = export_id, trajectory_id
        if not export_id or not trajectory_id:
            return []
        state, _ = reconcile(state, repository)
        return state.pinned

    selection = [Input("export", "value"), Input("trajectory", "value"), Input("threshold", "value"), Input("metric", "value"),
                 Input("checkpoint", "value")]

    def ready(*values):
        if any(v is None for v in values):
            raise dash.exceptions.PreventUpdate

    @app.callback(Output("overview-graph", "figure"), Output("checkpoint-summary", "children"), Output("checkpoint-label", "children"),
                  *selection, Input("axis", "value"), Input("log-x", "value"))
    @logged("overview")
    def overview(export_id, trajectory_id, threshold, metric, index, axis, log_x):
        ready(export_id, trajectory_id, threshold)
        state = current(export_id, trajectory_id, threshold, metric, index, axis=axis, log_x=log_x)
        rows = repository.select(export_id, "checkpoints", trajectory_id)
        units = repository.select(export_id, "behavior", trajectory_id)
        label = f"Clean behavioral score ({units.units.iloc[0]})" if not units.empty and "units" in units else "Clean behavioral score"
        try:
            figure = figures.overview(rows, threshold, state.axis, state.log_x, state.step, behavior_label=label)
        except ValueError as exc:
            figure = figures.overview(rows, threshold, "step", state.log_x, state.step, behavior_label=label)
            figure.add_annotation(text=str(exc), showarrow=False, x=.5, y=1.05, xref="paper", yref="paper")
        summary = figures.checkpoint_summary(rows, threshold)
        row = summary[summary.step == state.step]
        details = [] if row.empty else [html.Div(f"{k}: {v}") for k, v in row.iloc[0].items()
                                         if k not in {"run_id", "trajectory_id", "dataset_sha256"}]
        steps = repository.steps(export_id, trajectory_id)
        position = f"checkpoint {steps.index(state.step)+1}/{len(steps)}" if state.step in steps else ""
        return figure, [html.H4(f"Step {state.step}")] + details, f"Selected: step {state.step} ({position})"

    @app.callback(Output("ranking", "data"), Output("ranking", "columns"), Output("trajectories", "figure"), Output("heatmap", "figure"),
                  *selection, Input("pinned", "value"), Input("rank-mode", "value"), Input("axis", "value"), Input("log-x", "value"))
    @logged("components")
    def components_view(export_id, trajectory_id, threshold, metric, index, pinned, rank_mode, axis, log_x):
        ready(export_id, trajectory_id, threshold, metric)
        state = current(export_id, trajectory_id, threshold, metric, index, pinned, axis, log_x)
        rows = repository.select(export_id, "components", trajectory_id, threshold=threshold)
        table = figures.ranking(rows, state.step, metric, top=None)
        columns = [{"name": c, "id": c} for c in table.columns]
        try:
            trajectories = figures.component_trajectories(rows, metric, state.pinned, state.axis, state.log_x, bool(rank_mode))
        except ValueError:
            trajectories = figures.component_trajectories(rows, metric, state.pinned, "step", state.log_x, bool(rank_mode))
        return table.to_dict("records"), columns, trajectories, figures.heatmap(rows, metric)

    @app.callback(Output("distribution", "figure"), Output("dist-summary", "data"), *selection,
                  Input("dist-steps", "value"), Input("dist-mode", "value"))
    @logged("distributions")
    def distributions(export_id, trajectory_id, threshold, metric, index, extra, mode):
        ready(export_id, trajectory_id, threshold, metric)
        state = current(export_id, trajectory_id, threshold, metric, index)
        rows = repository.select(export_id, "components", trajectory_id, threshold=threshold)
        steps = list(dict.fromkeys([state.step, *(extra or [])]))
        figure, summary = figures.distribution(rows, steps, metric, mode)
        return figure, summary.round(6).to_dict("records")

    @functools.lru_cache(maxsize=int(config.get("cache_entries", 32)))
    def role_rows(export_id, manifest_sha, trajectory_id, threshold, quantile):
        """Export roles at their saved quantile; other quantiles use the shared role definition."""
        entry = repository.entry(export_id)
        saved = (entry["manifest"].get("roles") or {}).get("quantile")
        if saved is not None and abs(saved-quantile) < 1e-12:
            return repository.select(export_id, "roles", trajectory_id, threshold=threshold)
        from nnets_transformer.analysis.longitudinal import role_dynamics
        rows = repository.select(export_id, "components", trajectory_id, threshold=threshold).copy()
        rows["run_id"] = rows.trajectory_id
        return role_dynamics(rows, quantile)

    @app.callback(Output("roles-graph", "figure"), Output("membership", "children"), Output("similarity", "figure"),
                  *selection, Input("quantile", "value"), Input("log-x", "value"))
    @logged("roles")
    def roles(export_id, trajectory_id, threshold, metric, index, quantile, log_x):
        ready(export_id, trajectory_id, threshold, metric)
        state = current(export_id, trajectory_id, threshold, metric, index, log_x=log_x, quantile=quantile)
        rows = role_rows(export_id, repository.entry(export_id)["manifest_sha256"], trajectory_id, threshold, float(quantile))
        change = figures.membership_change(rows, metric, state.step)
        text = (f"Step {state.step} vs {change['previous_step']} (quantile {quantile}, strictly above; ties excluded)\n"
                f"Entering: {', '.join(change['entering']) or '—'}\nLeaving: {', '.join(change['leaving']) or '—'}\n"
                f"Retained: {', '.join(change['retained']) or '—'}")
        components = repository.select(export_id, "components", trajectory_id, threshold=threshold)
        return figures.roles(rows, metric, log_x=state.log_x), text, figures.similarity_matrix(components, metric, float(quantile))

    @app.callback(Output("compare-graph", "figure"), Output("behavior-facets", "figure"), Output("compare-notes", "children"),
                  Input("export", "value"), Input("trajectory", "value"), Input("threshold", "value"), Input("compare", "value"),
                  Input("compare-column", "value"), Input("axis", "value"), Input("log-x", "value"))
    @logged("compare")
    def compare(export_id, trajectory_id, threshold, others, column, axis, log_x):
        ready(export_id, trajectory_id, threshold)
        frame = repository.trajectories(export_id).set_index("trajectory_id")
        reference = frame.loc[trajectory_id]
        series, notes, behavior = {}, [], {}
        for trajectory in [trajectory_id, *(others or [])]:
            meta = frame.loc[trajectory]
            if (meta.get("graph_version"), meta.get("resolution")) != (reference.get("graph_version"), reference.get("resolution")):
                notes.append(f"Excluded {_trajectory_label({**meta.to_dict(), 'trajectory_id': trajectory})}: graph version/resolution differs.")
                continue
            rows = repository.select(export_id, "checkpoints", trajectory)
            rows = rows[rows.threshold == threshold].sort_values("step")
            if rows.empty:
                notes.append(f"Excluded {trajectory[:10]}: threshold {threshold} not precomputed.")
                continue
            name = f"{meta.get('model_id')} · {meta.get('task')} · {meta.get('replacement')}"
            series[name] = rows
            units = repository.select(export_id, "behavior", trajectory)
            unit = units.units.iloc[0] if not units.empty and "units" in units else "score units"
            behavior.setdefault(meta.get("task"), (unit, {}))[1][name] = rows
        log = bool(log_x and "log" in log_x)
        try:
            overlay = figures.compare(series, column, axis or "step", log)
            facets = figures.behavior_facets(behavior, axis or "step", log)
        except ValueError as exc:
            notes.append(f"{exc}; using training steps.")
            overlay, facets = figures.compare(series, column, "step", log), figures.behavior_facets(behavior, "step", log)
        notes.append("Component/head identities are linked within one run only; trajectories are aligned by training axis, not by component.")
        return overlay, facets, "\n".join(notes)

    @app.callback(Output("network", "elements"), Output("network-counts", "children"), Output("node-details", "children"),
                  *selection, Input("min-strength", "value"), Input("max-edges", "value"), Input("neighborhood", "value"),
                  Input("compare-step", "value"), Input("network", "tapNodeData"))
    @logged("network")
    def network(export_id, trajectory_id, threshold, metric, index, min_strength, max_edges, neighborhood, compare_step, tapped):
        ready(export_id, trajectory_id, threshold, metric)
        state = current(export_id, trajectory_id, threshold, metric, index)
        if not repository.available(export_id, "edges"):
            return [], "Network unavailable: this export has no endpoint-identified edges.", ""
        edges = repository.select(export_id, "edges", trajectory_id, threshold=threshold)
        nodes_rows = repository.select(export_id, "components", trajectory_id, threshold=threshold, step=state.step)
        values = nodes_rows[nodes_rows.metric == metric].set_index("component").value.to_dict()
        nodes = sorted(nodes_rows.component.unique())
        inventory = repository.select(export_id, "components", trajectory_id).drop_duplicates("component")[["component", "layer", "head"]]
        positions = figures.layout(inventory)
        current_edges = edges[edges.step == state.step]
        label = f"step {state.step}"
        if compare_step is not None and compare_step != state.step:
            current_edges = figures.edge_difference(edges[edges.step == compare_step], current_edges)
            label = f"difference: step {compare_step} (red, only A) vs step {state.step} (green, only B)"
        selected = (tapped or {}).get("id")
        elements, counts = figures.network_elements(nodes, current_edges, positions, values, selected,
            bool(neighborhood), min_strength, int(max_edges) if max_edges else None)
        text = (f"{label}: displaying {counts['nodes_displayed']}/{counts['nodes_total']} nodes and "
                f"{counts['edges_displayed']}/{counts['edges_total']} edges ({counts['edges_after_filters']} after filters)"
                f"{' — TRUNCATED display; full-graph metrics unchanged' if counts['truncated'] else ''}")
        details = ""
        if selected:
            rows = nodes_rows[nodes_rows.component == selected]
            details = f"{selected}\n" + "\n".join(f"{r.metric}: {r.value}" for r in rows.itertuples())
        return elements, text, details

    @app.callback(Output("status-table", "data"), Input("status-interval", "n_intervals"), Input("refresh", "n_clicks"))
    @logged("status")
    def status(_tick, _refresh):
        return attempts(config.get("log_root", "logs"), limit=int(config.get("status_attempts", 100)))

    @app.callback(Output("log-tail", "children"), Input("status-table", "selected_rows"), Input("severity", "value"),
                  Input("status-interval", "n_intervals"), State("status-table", "data"))
    @logged("log_tail")
    def log_tail(selected, severity, _tick, rows):
        if not selected or not rows:
            return "Select an attempt to show its bounded console tail."
        row = rows[selected[0]]
        resume = f"\nResume: {row.get('resume_command')}" if row.get("resume_command") else ""
        return f"{row['directory']}{resume}\n\n" + tail(row["directory"], max_bytes=int(config.get("log_tail_bytes", 20000)), severity=severity)

    @app.callback(Output("share-link", "value"), *selection, Input("axis", "value"), Input("log-x", "value"))
    @logged("share")
    def share(export_id, trajectory_id, threshold, metric, index, axis, log_x):
        ready(export_id, trajectory_id)
        return current(export_id, trajectory_id, threshold, metric, index, axis=axis, log_x=log_x).query()

    @app.callback(Output("download", "data"), Input("export-view", "n_clicks"), Input("export-chart", "n_clicks"),
                  Input("export-ranking", "n_clicks"), Input("export-table", "n_clicks"), *[State(i.component_id, i.component_property) for i in selection],
                  State("pinned", "value"), State("axis", "value"), State("log-x", "value"), State("quantile", "value"),
                  State("compare", "value"), State("overview-graph", "figure"), prevent_initial_call=True)
    @logged("export")
    def download(_v, _c, _r, _t, export_id, trajectory_id, threshold, metric, index, pinned, axis, log_x, quantile, others, figure):
        ready(export_id, trajectory_id, ctx.triggered_id)
        state = current(export_id, trajectory_id, threshold, metric, index, pinned, axis, log_x, quantile)
        state.compare = others or []
        meta = provenance(repository, state)
        meta["view"] = json.loads(state.to_json())["state"]
        if job is not None:
            job.event("dashboard_export", kind=ctx.triggered_id, export_id=export_id, trajectory_id=trajectory_id, step=state.step)
        name = f"{repository.entry(export_id)['name']}-step{state.step}"
        if ctx.triggered_id == "export-view":
            return dict(content=state.to_json(meta), filename=f"{name}-view.json")
        if ctx.triggered_id == "export-chart":
            import plotly.graph_objects as go
            return dict(content=chart_html(go.Figure(figure), meta, f"Overview {name}"), filename=f"{name}-overview.html")
        if ctx.triggered_id == "export-ranking":
            rows = repository.select(export_id, "components", trajectory_id, threshold=threshold)
            table = figures.ranking(rows, state.step, metric, top=None)
            return dcc.send_bytes(table_bytes(table, "csv", meta), f"{name}-{metric}-ranking.csv")
        rows = figures.checkpoint_summary(repository.select(export_id, "checkpoints", trajectory_id), threshold)
        return dcc.send_bytes(table_bytes(rows, "parquet", meta), f"{name}-checkpoints.parquet")

    return app
